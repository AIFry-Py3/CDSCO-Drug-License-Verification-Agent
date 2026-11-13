"""OCR tool: EasyOCR (default, pip only) or Tesseract (OCR_ENGINE=tesseract, needs system binary)."""
import os, re, json, time
import numpy as np
from dotenv import load_dotenv
load_dotenv()

LOW = 0.6          # average-confidence threshold from the plan
_reader = None

def _lines(img):
    """Returns [(text, confidence 0-1)] per line."""
    global _reader
    if os.getenv("OCR_ENGINE", "easyocr") == "tesseract":
        import pytesseract
        d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        g = {}
        for t, c, b, p, l in zip(d["text"], d["conf"], d["block_num"], d["par_num"], d["line_num"]):
            if t.strip() and float(c) >= 0: g.setdefault((b, p, l), []).append((t, float(c) / 100))
        return [(" ".join(t for t, _ in v), sum(c for _, c in v) / len(v)) for v in g.values()]
    import easyocr
    if _reader is None: _reader = easyocr.Reader(["en"], gpu=False)
    return [(t, float(c)) for _, t, c in _reader.readtext(img)]

def preprocess(img):
    """grayscale -> 2x upscale -> Otsu threshold (plan's retry recipe)."""
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def read(image_bytes):
    import cv2
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Could not decode image")
    tries = []
    for label, im in [("raw", img), ("preprocessed", None)]:
        im = preprocess(img) if im is None else im
        ls = _lines(im)
        conf = sum(c for _, c in ls) / len(ls) if ls else 0.0
        tries.append({"attempt": label, "conf": round(conf, 2), "text": "\n".join(t for t, _ in ls)})
        if conf >= LOW: break
    best = max(tries, key=lambda t: t["conf"])
    return {"text": best["text"], "conf": best["conf"], "attempts": [(t["attempt"], t["conf"]) for t in tries]}

_NAME = re.compile(r"\b(ltd|limited|pvt|private|pharma\w*|labs?|laborator\w+|industries|drugs?|remedies|healthcare)\b", re.I)
_PREFIX = re.compile(r"^\s*(mfd\.?|manufactured)\s*(by)?\s*[:\-]?\s*", re.I)

def _heuristic(text):
    ls = [_PREFIX.sub("", l).strip() for l in text.splitlines() if l.strip()]
    if not ls: return {"name": "", "address": ""}
    name = next((l for l in ls if _NAME.search(l)), ls[0])
    return {"name": name, "address": ", ".join(l for l in ls if l != name)}

def extract(text):
    """Pull name/address out of OCR text: Gemini if key set, else heuristic."""
    if os.getenv("GEMINI_API_KEY"):
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        prompt = ("From this OCR text of a drug label/document, extract the MANUFACTURER name and its address "
                  "(fix obvious OCR errors, ignore other text). Reply JSON {\"name\": str, \"address\": str}.\n\n" + text)
        for a in range(3):
            try:
                r = client.models.generate_content(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), contents=prompt,
                                                   config={"response_mime_type": "application/json"})
                d = json.loads(r.text)
                return {"name": d.get("name", ""), "address": d.get("address", "")}
            except Exception: time.sleep(2 ** a)
    return _heuristic(text)
