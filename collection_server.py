"""
MTG collection server: user accounts + per-user card collections in SQLite.

Setup:   pip install fastapi uvicorn requests
Run:     uvicorn collection_server:app --reload
Docs:    http://127.0.0.1:8000/docs   (interactive API page)

Auth is bearer-token based: /register or /login returns a token, and you send
it as   Authorization: Bearer <token>   on every other request.
"""
import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import Optional

import requests
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

DB_PATH = os.environ.get("MTG_DB", "collection.db")
TOKEN_TTL = 30 * 24 * 3600  # tokens last 30 days
PRICE_MAX_AGE = 24 * 3600   # Scryfall updates prices daily, so refresh at least that often
SCRYFALL_COLLECTION_URL = "https://api.scryfall.com/cards/collection"
SCRYFALL_HEADERS = {
    "User-Agent": "MTGCollectionServer/0.1",
    "Accept": "application/json;q=0.9,*/*;q=0.8",
}

app = FastAPI(title="MTG Collection")


# ------------------------------------------------------------------ database
def connect():
    """Every connection goes through here. The timeout makes a writer wait for
    another writer to finish instead of failing straight away."""
    conn = sqlite3.connect(DB_PATH, timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = connect()
    # WAL mode lets readers and a writer work at the same time (setting is saved in the DB file)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            pw_hash TEXT NOT NULL,
            created REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tokens (
            token_hash TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cards (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            scryfall_id TEXT NOT NULL,
            name TEXT NOT NULL,
            set_code TEXT NOT NULL,
            collector_number TEXT NOT NULL,
            foil INTEGER NOT NULL DEFAULT 0,
            quantity INTEGER NOT NULL DEFAULT 1,
            added REAL NOT NULL,
            UNIQUE (user_id, scryfall_id, foil)
        );
        -- One row per card printing, shared by all users. Prices are integer US
        -- cents (avoids float rounding); NULL means Scryfall lists no price.
        CREATE TABLE IF NOT EXISTS prices (
            scryfall_id TEXT PRIMARY KEY,
            usd INTEGER,
            usd_foil INTEGER,
            usd_etched INTEGER,
            updated REAL NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


init_db()


def get_db():
    conn = connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ------------------------------------------------------------------ security
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"{salt.hex()}:{h.hex()}"


def verify_password(password: str, stored: str) -> bool:
    salt_hex, h_hex = stored.split(":")
    h = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1)
    return hmac.compare_digest(h.hex(), h_hex)


DUMMY_HASH = hash_password("dummy-password")  # keeps login timing similar for unknown users


def issue_token(db, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    # Only a hash of the token is stored, so a leaked database can't be used to log in
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    db.execute(
        "INSERT INTO tokens (token_hash, user_id, expires) VALUES (?, ?, ?)",
        (token_hash, user_id, time.time() + TOKEN_TTL),
    )
    return token


def current_user(authorization: Optional[str] = Header(None), db=Depends(get_db)) -> int:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing bearer token")
    token_hash = hashlib.sha256(authorization[7:].encode()).hexdigest()
    row = db.execute(
        "SELECT user_id FROM tokens WHERE token_hash = ? AND expires > ?",
        (token_hash, time.time()),
    ).fetchone()
    if not row:
        raise HTTPException(401, "Invalid or expired token")
    return row["user_id"]


# -------------------------------------------------------------------- prices
def to_cents(value) -> Optional[int]:
    """'1.50' -> 150, None -> None."""
    try:
        return int(Decimal(value) * 100)
    except (InvalidOperation, TypeError):
        return None


def refresh_prices(scryfall_ids) -> int:
    """Fetch current prices from Scryfall and store them. Returns how many were updated.
    Opens its own connection so it can run as a background task."""
    ids = list(dict.fromkeys(scryfall_ids))  # de-duplicate, keep order
    if not ids:
        return 0
    conn = connect()
    updated = 0
    try:
        for i in range(0, len(ids), 75):  # the endpoint accepts up to 75 cards per request
            batch = ids[i:i + 75]
            try:
                r = requests.post(
                    SCRYFALL_COLLECTION_URL,
                    json={"identifiers": [{"id": x} for x in batch]},
                    headers=SCRYFALL_HEADERS, timeout=15,
                )
                r.raise_for_status()
            except requests.RequestException:
                break  # keep the old prices and try again on the next refresh
            for card in r.json().get("data", []):
                pr = card.get("prices", {})
                conn.execute(
                    """INSERT INTO prices (scryfall_id, usd, usd_foil, usd_etched, updated)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT (scryfall_id) DO UPDATE SET
                         usd = excluded.usd, usd_foil = excluded.usd_foil,
                         usd_etched = excluded.usd_etched, updated = excluded.updated""",
                    (card["id"], to_cents(pr.get("usd")), to_cents(pr.get("usd_foil")),
                     to_cents(pr.get("usd_etched")), time.time()),
                )
                updated += 1
            conn.commit()
            time.sleep(0.1)  # Scryfall asks for 50-100 ms between requests
    finally:
        conn.close()
    return updated


def stale_price_ids(db, user_id: int):
    rows = db.execute(
        """SELECT DISTINCT c.scryfall_id FROM cards c
           LEFT JOIN prices p ON p.scryfall_id = c.scryfall_id
           WHERE c.user_id = ? AND (p.scryfall_id IS NULL OR p.updated < ?)""",
        (user_id, time.time() - PRICE_MAX_AGE),
    ).fetchall()
    return [r["scryfall_id"] for r in rows]


# A foil row is priced with the foil price (or etched), a normal row with the normal
# price. A foil never falls back to the non-foil price, so a missing price shows as
# missing instead of silently being wrong.
COLLECTION_SQL = """
    SELECT c.id, c.scryfall_id, c.name, c.set_code, c.collector_number, c.foil, c.quantity,
           CASE WHEN c.foil THEN COALESCE(p.usd_foil, p.usd_etched) ELSE p.usd END AS unit_cents,
           p.updated AS price_updated
    FROM cards c LEFT JOIN prices p ON p.scryfall_id = c.scryfall_id
    WHERE c.user_id = ?
    ORDER BY c.name COLLATE NOCASE, c.set_code
"""


# -------------------------------------------------------------------- models
class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)


class CardIn(BaseModel):
    scryfall_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=200)
    set_code: str = Field(min_length=1, max_length=10)
    collector_number: str = Field(min_length=1, max_length=20)
    foil: bool = False
    quantity: int = Field(1, ge=1, le=999)


# ----------------------------------------------------------------- endpoints
@app.post("/register")
def register(creds: Credentials, db=Depends(get_db)):
    try:
        cur = db.execute(
            "INSERT INTO users (username, pw_hash, created) VALUES (?, ?, ?)",
            (creds.username, hash_password(creds.password), time.time()),
        )
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Username already taken")
    return {"token": issue_token(db, cur.lastrowid)}


@app.post("/login")
def login(creds: Credentials, db=Depends(get_db)):
    row = db.execute(
        "SELECT id, pw_hash FROM users WHERE username = ?", (creds.username,)
    ).fetchone()
    ok = verify_password(creds.password, row["pw_hash"] if row else DUMMY_HASH)
    if not row or not ok:
        raise HTTPException(401, "Wrong username or password")
    return {"token": issue_token(db, row["id"])}


@app.post("/cards")
def add_card(card: CardIn, background: BackgroundTasks,
             user_id: int = Depends(current_user), db=Depends(get_db)):
    """Add a card. Adding the same printing + finish again increases its quantity."""
    db.execute(
        """
        INSERT INTO cards (user_id, scryfall_id, name, set_code, collector_number,
                           foil, quantity, added)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (user_id, scryfall_id, foil)
        DO UPDATE SET quantity = quantity + excluded.quantity
        """,
        (user_id, card.scryfall_id, card.name, card.set_code.lower(),
         card.collector_number, int(card.foil), card.quantity, time.time()),
    )
    row = db.execute(
        "SELECT id, quantity FROM cards WHERE user_id = ? AND scryfall_id = ? AND foil = ?",
        (user_id, card.scryfall_id, int(card.foil)),
    ).fetchone()
    db.commit()  # release the write lock now; the background task writes on its own connection
    price = db.execute("SELECT updated FROM prices WHERE scryfall_id = ?",
                       (card.scryfall_id,)).fetchone()
    if not price or price["updated"] < time.time() - PRICE_MAX_AGE:
        background.add_task(refresh_prices, [card.scryfall_id])  # runs after the response
    return {"id": row["id"], "name": card.name, "quantity": row["quantity"]}


@app.get("/collection")
def list_collection(user_id: int = Depends(current_user), db=Depends(get_db)):
    out = []
    for r in db.execute(COLLECTION_SQL, (user_id,)).fetchall():
        cents = r["unit_cents"]
        out.append({
            "id": r["id"], "scryfall_id": r["scryfall_id"], "name": r["name"],
            "set_code": r["set_code"], "collector_number": r["collector_number"],
            "foil": bool(r["foil"]), "quantity": r["quantity"],
            "unit_price_usd": None if cents is None else cents / 100,
            "price_updated": r["price_updated"],  # unix time; None if never fetched
        })
    return out


@app.get("/collection/value")
def collection_value(user_id: int = Depends(current_user), db=Depends(get_db)):
    rows = db.execute(COLLECTION_SQL, (user_id,)).fetchall()
    priced = [r for r in rows if r["unit_cents"] is not None]
    return {
        "total_usd": sum(r["unit_cents"] * r["quantity"] for r in priced) / 100,
        "total_cards": sum(r["quantity"] for r in rows),
        "entries_without_price": len(rows) - len(priced),
        "oldest_price_update": min((r["price_updated"] for r in priced), default=None),
    }


@app.post("/prices/refresh")
def refresh_my_prices(force: bool = False, user_id: int = Depends(current_user),
                      db=Depends(get_db)):
    """Refresh missing/stale prices for your cards (force=true refreshes all of them).
    Call this from a daily cron job or a button."""
    if force:
        rows = db.execute("SELECT DISTINCT scryfall_id FROM cards WHERE user_id = ?",
                          (user_id,)).fetchall()
        ids = [r["scryfall_id"] for r in rows]
    else:
        ids = stale_price_ids(db, user_id)
    return {"requested": len(ids), "updated": refresh_prices(ids)}


@app.delete("/cards/{card_id}")
def delete_card(card_id: int, user_id: int = Depends(current_user), db=Depends(get_db)):
    cur = db.execute("DELETE FROM cards WHERE id = ? AND user_id = ?", (card_id, user_id))
    if cur.rowcount == 0:
        raise HTTPException(404, "Card not found")
    return {"deleted": card_id}


@app.get("/collection.txt", response_class=PlainTextResponse)
def export_text(
    fmt: str = Query("detailed", alias="format", pattern="^(detailed|simple|priced)$"),
    user_id: int = Depends(current_user),
    db=Depends(get_db),
):
    """
    Plain-text export, one line per card.
      detailed:  1 Lightning Bolt (M11) 146 *F*     (importable by Moxfield/Archidekt)
      simple:    4 Lightning Bolt                   (quantities merged across printings)
      priced:    2 Lightning Bolt (M11) 146  -  $1.50 ea, plus a total (for reading,
                 not for importing)
    """
    rows = db.execute(COLLECTION_SQL, (user_id,)).fetchall()

    def detailed(r):
        return (f"{r['quantity']} {r['name']} ({r['set_code'].upper()}) "
                f"{r['collector_number']}" + (" *F*" if r["foil"] else ""))

    if fmt == "simple":
        totals = defaultdict(int)
        for r in rows:
            totals[r["name"]] += r["quantity"]
        lines = [f"{qty} {name}" for name, qty in totals.items()]
    elif fmt == "priced":
        lines = [
            detailed(r) + ("  -  no price" if r["unit_cents"] is None
                           else f"  -  ${r['unit_cents'] / 100:.2f} ea")
            for r in rows
        ]
        total = sum(r["unit_cents"] * r["quantity"] for r in rows if r["unit_cents"] is not None)
        missing = sum(1 for r in rows if r["unit_cents"] is None)
        if rows:
            lines.append(f"Total: ${total / 100:.2f}"
                         + (f" ({missing} entries without a price)" if missing else ""))
    else:
        lines = [detailed(r) for r in rows]
    return "\n".join(lines) + ("\n" if lines else "")
