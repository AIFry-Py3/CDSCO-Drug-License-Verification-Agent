"""Shared Gemini JSON call. Paces calls per model (free tier = 5 RPM), waits out per-minute 429s,
marks a model dead only on per-day 429s, rotates keys x models, returns None on total failure."""
import os, re, json, time
from google import genai
from google.genai import errors

_dead, _last = set(), {}
GAP = float(os.getenv("GEMINI_MIN_GAP", "13"))  # seconds between calls to the same model

def _keys():
    return [v for k, v in sorted(os.environ.items()) if k.startswith("GEMINI_API_KEY") and v]

def _models():
    d = os.getenv("GEMINI_MODEL", "gemini-3.5-flash") + "," + os.getenv("GEMINI_FALLBACK", "gemini-3.5-flash-lite")
    return [m.strip() for m in os.getenv("GEMINI_MODELS", d).split(",") if m.strip()]

def _slots():
    return [(ki, k, m) for ki, k in enumerate(_keys()) for m in _models() if (ki, m) not in _dead]

def generate_json(prompt, tries=6):
    clients = {}
    for _ in range(tries):
        slots = _slots()
        if not slots: return None
        now = time.time()
        ready = [s for s in slots if now - _last.get((s[0], s[2]), 0) >= GAP]
        if not ready:  # all paced out: wait for the earliest
            time.sleep(max(0.5, min(GAP - (now - _last[(s[0], s[2])]) for s in slots))); continue
        ki, key, m = ready[0]
        _last[(ki, m)] = time.time()
        c = clients.setdefault(key, genai.Client(api_key=key, http_options={"timeout": 30000}))
        try:
            r = c.models.generate_content(model=m, contents=prompt,
                                          config={"response_mime_type": "application/json"})
            return json.loads(r.text)
        except errors.ClientError as e:
            msg = str(e)
            if e.code == 429 and "PerDay" in msg: _dead.add((ki, m))
            elif e.code == 429:
                w = re.search(r"retry in ([\d.]+)s", msg)
                _last[(ki, m)] = time.time() + (float(w.group(1)) if w else GAP) - GAP
            print("gemini", m, e.code, "per-day" if "PerDay" in msg else "")
        except Exception as e:
            print("gemini", m, type(e).__name__); time.sleep(1)
    return None
