"""Human-in-the-loop memory. Confirmed decisions are checked FIRST on future runs."""
import json, os, datetime
import backend as B

PATH = "data/feedback.json"

def _key(name, address):
    qn, qa, qp = B.core(name), B.norm(address), B.pin(address)
    return f"{qn}|{qp or qa}"

def _load():
    return json.load(open(PATH)) if os.path.exists(PATH) else {}

def lookup(name, address):
    return _load().get(_key(name, address))

def save(name, address, verdict, candidate=None):
    """verdict: APPROVED (pass the matched candidate dict) or REJECTED (not listed)."""
    assert verdict in ("APPROVED", "REJECTED")
    d, c = _load(), candidate or {}
    d[_key(name, address)] = {"verdict": verdict, "name": c.get("name"), "address": c.get("address"),
                              "license_no": c.get("license_no"), "valid_upto": c.get("valid_upto", ""),
                              "at": datetime.datetime.now().isoformat(timespec="seconds")}
    os.makedirs(os.path.dirname(PATH) or ".", exist_ok=True)
    json.dump(d, open(PATH, "w"), indent=1)
