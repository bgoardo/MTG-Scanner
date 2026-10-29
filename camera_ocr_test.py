"""
Webcam OCR test for an MTG card scanner.

Controls (click the preview window first):
  r      draw a box around the card's title bar (drag, then ENTER; C to cancel)
  SPACE  run OCR on the box, then look the name up on Scryfall
         (with no box set, OCRs the whole frame and skips the lookup)
  f      run OCR on the whole frame (no Scryfall lookup)
  a      add the matched card to your collection (Shift+A adds it as foil)
  e      download your collection as plain text to collection.txt
  s      save the current frame and the last OCR crop to ./captures
  q/ESC  quit

Collection (optional): start collection_server.py, then run this script with
  --username NAME   (add --register the first time to create the account)

Setup:
  pip install opencv-python pytesseract numpy requests
  Install the Tesseract program too:
    Windows: https://github.com/UB-Mannheim/tesseract/wiki
    macOS:   brew install tesseract
    Linux:   sudo apt install tesseract-ocr
"""
import argparse
import getpass
import os
import re
import time

import cv2
import pytesseract
import requests
from pytesseract import Output


def sharpness(gray):
    """Variance of the Laplacian: higher = sharper. Compare values between
    focus settings or camera heights rather than trusting an absolute number."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def run_ocr(img, psm):
    bw = preprocess(img)
    data = pytesseract.image_to_data(
        bw, config=f"--psm {psm}", output_type=Output.DICT
    )
    words, confs = [], []
    for text, conf in zip(data["text"], data["conf"]):
        if text.strip() and float(conf) >= 0:
            words.append(text.strip())
            confs.append(float(conf))
    avg = sum(confs) / len(confs) if confs else 0.0
    return " ".join(words), avg, bw


# ---------------------------------------------------------------- Scryfall
SCRYFALL_URL = "https://api.scryfall.com/cards/named"
HEADERS = {
    "User-Agent": "MTGCardScanner/0.1",   # Scryfall asks for a descriptive UA
    "Accept": "application/json;q=0.9,*/*;q=0.8",
}
_cache = {}          # cleaned OCR text -> result, avoids repeat requests
_last_call = 0.0


def clean_ocr_text(text):
    """Keep letters, apostrophes, commas, hyphens and spaces; drop OCR junk."""
    text = re.sub(r"[^\w',\- ]|_", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def scryfall_lookup(ocr_text):
    """Fuzzy-match OCR text to a card. Returns a dict with a 'status' of
    ok / fail / skip / error."""
    global _last_call
    query = clean_ocr_text(ocr_text)
    if len(query) < 3:
        return {"status": "skip", "message": "OCR text too short"}
    key = query.lower()
    if key in _cache:
        return _cache[key]

    # Be polite: Scryfall asks for roughly 50-100 ms between requests
    wait = 0.1 - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    try:
        r = requests.get(SCRYFALL_URL, params={"fuzzy": query},
                         headers=HEADERS, timeout=10)
        _last_call = time.time()
        data = r.json()
    except (requests.RequestException, ValueError) as e:
        return {"status": "error", "message": str(e)}   # not cached

    if r.status_code == 200:
        result = {"status": "ok", "query": query, "card": data}
    else:
        result = {"status": "fail", "query": query,
                  "message": data.get("details", f"HTTP {r.status_code}")}
    _cache[key] = result
    return result


def describe_card(card):
    prices = card.get("prices", {})
    usd = prices.get("usd") or prices.get("usd_foil") or prices.get("usd_etched")
    price = f"${usd}" if usd else "no price"
    return (f"{card['name']} [{card['set'].upper()} #{card['collector_number']}] "
            f"{price}")


def ascii_only(text):
    """cv2.putText can't draw accents/symbols, so strip them for the overlay."""
    return text.encode("ascii", "ignore").decode()


# ------------------------------------------------------- collection server
def api_login(server, username, password, register=False):
    path = "/register" if register else "/login"
    r = requests.post(f"{server}{path}",
                      json={"username": username, "password": password}, timeout=10)
    if r.status_code != 200:
        raise SystemExit(f"{path} failed: {r.json().get('detail', r.text)}")
    return r.json()["token"]


def api_add_card(server, token, card, foil):
    payload = {
        "scryfall_id": card["id"],
        "name": card["name"],
        "set_code": card["set"],
        "collector_number": card["collector_number"],
        "foil": foil,
        "quantity": 1,
    }
    r = requests.post(f"{server}/cards", json=payload, timeout=10,
                      headers={"Authorization": f"Bearer {token}"})
    if r.status_code != 200:
        try:
            return False, r.json().get("detail", r.text)
        except ValueError:
            return False, f"server error {r.status_code}"
    return True, f"{'Foil ' if foil else ''}{r.json()['name']}: now x{r.json()['quantity']}"


def api_export_text(server, token):
    r = requests.get(f"{server}/collection.txt", timeout=10,
                     headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    return r.text


def open_camera(index, width, height, no_dshow):
    if os.name == "nt" and not no_dshow:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--no-dshow", action="store_true", help="skip DirectShow on Windows")
    ap.add_argument("--tesseract-cmd", help=r"e.g. C:\Program Files\Tesseract-OCR\tesseract.exe")
    ap.add_argument("--server", default="http://127.0.0.1:8000", help="collection server URL")
    ap.add_argument("--username", help="log in to the collection server as this user")
    ap.add_argument("--register", action="store_true", help="create the account first")
    ap.add_argument("--no-lookup", action="store_true", help="OCR only, skip Scryfall")
    ap.add_argument("--no-autofocus", action="store_true", help="try to disable autofocus")
    args = ap.parse_args()

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    token = None
    if args.username:
        password = getpass.getpass(f"Password for {args.username}: ")
        try:
            token = api_login(args.server, args.username, password, args.register)
        except requests.RequestException as e:
            raise SystemExit(f"Could not reach {args.server}: {e}")
        print(f"Logged in as {args.username}")

    cap = open_camera(args.camera, args.width, args.height, args.no_dshow)
    if not cap.isOpened():
        raise SystemExit("Could not open camera. Try a different --camera index.")
    if args.no_autofocus:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera running at {actual_w}x{actual_h}")
    print(__doc__)

    roi = None            # (x, y, w, h)
    last_text, last_conf = "", 0.0
    card_line, card_ok = "", False
    last_card, status = None, ""
    last_crop = None
    os.makedirs("captures", exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp_all = sharpness(gray)

        view = frame.copy()
        if roi:
            x, y, w, h = roi
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
            sharp_roi = sharpness(gray[y:y + h, x:x + w])
            hud = f"sharp frame {sharp_all:.0f} | title box {sharp_roi:.0f}"
        else:
            hud = f"sharp frame {sharp_all:.0f} | no box set (press r)"

        cv2.putText(view, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        if last_text:
            cv2.putText(view, ascii_only(f"OCR: {last_text} ({last_conf:.0f}%)"), (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if card_line:
            color = (0, 255, 0) if card_ok else (0, 0, 255)
            cv2.putText(view, ascii_only(card_line), (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if status:
            cv2.putText(view, ascii_only(status), (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)

        # Scale down big frames so the preview fits on screen
        scale = min(1.0, 1280 / view.shape[1])
        if scale < 1.0:
            view = cv2.resize(view, None, fx=scale, fy=scale)
        cv2.imshow("MTG OCR test", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

        elif key == ord("r"):
            sel = cv2.selectROI("Select title bar, then ENTER", frame, showCrosshair=False)
            cv2.destroyWindow("Select title bar, then ENTER")
            if sel[2] > 0 and sel[3] > 0:
                roi = tuple(int(v) for v in sel)

        elif key in (ord(" "), ord("f")):
            is_title = key == ord(" ") and roi is not None
            if is_title:
                x, y, w, h = roi
                crop, psm = frame[y:y + h, x:x + w], 7   # 7 = single line of text
            else:
                crop, psm = frame, 6                     # 6 = block of text
            t0 = time.time()
            last_text, last_conf, bw = run_ocr(crop, psm)
            last_crop = bw
            print(f"[{time.time() - t0:.2f}s] '{last_text}'  confidence {last_conf:.0f}%")
            cv2.imshow("What OCR saw", bw)

            card_line, card_ok, last_card, status = "", False, None, ""
            if is_title and not args.no_lookup:
                res = scryfall_lookup(last_text)
                if res["status"] == "ok":
                    card = res["card"]
                    card_line, card_ok = describe_card(card), True
                    last_card = card
                    print(f"  Scryfall: {card_line}")
                    print(f"  (searched for '{res['query']}', "
                          f"{card.get('type_line', '')}, {card['set_name']})")
                else:
                    card_line = f"No match: {res['message']}"[:70]
                    print(f"  Scryfall {res['status']}: {res['message']}")

        elif key in (ord("a"), ord("A")):
            if not token:
                status = "Not logged in (use --username)"
            elif not last_card:
                status = "Scan a card first (SPACE)"
            else:
                try:
                    ok, msg = api_add_card(args.server, token, last_card, foil=(key == ord("A")))
                    status = ("Added " if ok else "Failed: ") + msg
                except requests.RequestException as e:
                    status = "Server unreachable"
                    print(e)
            print(status)

        elif key == ord("e"):
            if not token:
                status = "Not logged in (use --username)"
            else:
                try:
                    text = api_export_text(args.server, token)
                    with open("collection.txt", "w", encoding="utf-8") as f:
                        f.write(text)
                    status = f"Saved collection.txt ({len(text.splitlines())} lines)"
                    print(text)
                except requests.RequestException as e:
                    status = "Export failed"
                    print(e)

        elif key == ord("s"):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"captures/{stamp}_frame.png", frame)
            if last_crop is not None:
                cv2.imwrite(f"captures/{stamp}_ocr_crop.png", last_crop)
            print(f"Saved to captures/{stamp}_*.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
