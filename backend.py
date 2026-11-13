import os, re, json, time, datetime
import pandas as pd
from rapidfuzz import fuzz, process
from dotenv import load_dotenv
load_dotenv()

DATA = "data/records.csv"
ABBR = {"pvt": "private", "ltd": "limited", "labs": "laboratories", "co": "company",
        "inds": "industries", "mfg": "manufacturing"}
GENERIC = {"private", "limited", "company", "laboratories", "laboratory", "pharmaceuticals",
           "pharmaceutical", "pharma", "industries", "drugs", "and", "india", "manufacturing"}
APPROVE, AMBIG = 90, 70   # thresholds from the plan - tune on your test set

def norm(s):
    s = re.sub(r"[^a-z0-9\s]", " ", str(s or "").lower().replace("&", " and "))
    return " ".join(ABBR.get(w, w) for w in s.split())

LEGAL = {"private", "limited", "company"}
def core(s):  # drop legal-form words so "Private Limited" doesn't inflate scores
    return " ".join(w for w in norm(s).split() if w not in LEGAL)

def pin(s):
    m = re.search(r"\b\d{6}\b", str(s or ""))
    return m.group(0) if m else ""

def prep(df):
    df = df.fillna("").astype(str).reset_index(drop=True)
    df["name_n"] = df["name"].map(core)
    df["addr_n"] = df["address"].map(norm)
    df["full_n"] = df["name_n"] + " " + df["addr_n"]
    df["pin"] = df["address"].map(pin)
    return df

def load(path=DATA):
    return prep(pd.read_csv(path, dtype=str))

def score(a, b, **kw):
    # token_set alone over-scores shared words ("private limited") and subsets; blend with token_sort
    return (fuzz.token_set_ratio(a, b) + fuzz.token_sort_ratio(a, b)) / 2

def top(df, query, col, k=5):
    if df.empty or not query: return []
    res = process.extract(query, df[col].tolist(), scorer=score, limit=k)
    return [(int(df.index[i]), float(s)) for _, s, i in res]   # (row label, score)

def compare(row, qa, qp):
    pm = (row["pin"] == qp) if (qp and row["pin"]) else None
    asc = fuzz.partial_ratio(qa, row["addr_n"]) if qa else None
    if pm is not None: ok = pm
    elif asc is not None: ok = asc >= 70
    else: ok = None
    return {"pin_match": pm, "addr_score": asc, "consistent": ok}

def rewrite_plans(df, qn, qa, qp):
    core = " ".join(w for w in qn.split() if w not in GENERIC) or qn
    yield "drop generic words", df, core, "name_n"
    yield "name + address", df, f"{core} {qa}".strip(), "full_n"
    if qp: yield "search by PIN", df[df["pin"] == qp], core, "name_n"

def llm_verify(name, address, c):
    if not os.getenv("GEMINI_API_KEY"): return None
    prompt = (f"Query manufacturer: {name} | {address}\nCandidate in licensed list: {c['name']} | {c['address']}\n"
              "Is the candidate the same legal entity at the same location (allow typos/abbreviations, "
              "but different city/PIN = different)? Reply JSON: {\"same_entity\": bool, \"reason\": str}")
    from gem import generate_json
    return generate_json(prompt)

def verify(name, address="", df=None, use_llm=True, log_dir="logs"):
    df = load() if df is None else df
    trace = []
    def log(step, **kw): trace.append({"t": round(time.time(), 2), "step": step, **kw})
    qn, qa, qp = core(name), norm(address), pin(address)
    log("normalize", name=qn, address=qa, pin=qp)

    cands, via = top(df, qn, "name_n"), "direct"
    log("fuzzy_name", top=[(df.loc[i, "name"], s) for i, s in cands])
    if not cands or cands[0][1] < AMBIG:
        for n, (label, sub, q, col) in enumerate(rewrite_plans(df, qn, qa, qp), 1):
            c2 = top(sub, q, col)
            log(f"rewrite_{n}", plan=label, query=q, top=[(df.loc[i, "name"], s) for i, s in c2])
            if c2 and c2[0][1] >= AMBIG: cands, via = c2, label; break

    evald = []
    for i, s in cands:
        r = df.loc[i]
        evald.append({"name": r["name"], "address": r["address"], "license_no": r["license_no"],
                      "state": r.get("state", ""), "score": s, **compare(r, qa, qp)})
    if evald: log("address_compare", candidates=[(e["name"], e["score"], e["pin_match"], e["addr_score"]) for e in evald])
    plausible = [e for e in evald if e["score"] >= AMBIG]
    best = max(plausible, key=lambda e: (e["consistent"] is True, e["score"])) if plausible else (evald[0] if evald else None)
    score = best["score"] if best else 0

    if best and score >= AMBIG:
        if via == "direct" and score >= APPROVE and best["consistent"] is True:
            verdict, why = "APPROVED", "Strong name match and address/PIN consistent."
        else:
            verdict = "NEEDS HUMAN REVIEW"
            why = (f"Matched via '{via}'. " if via != "direct" else "") + \
                  ("Name matches but address/PIN inconsistent or missing." if score >= APPROVE else "Name match is ambiguous.")
        conf = round(score / 100, 2)
    else:
        verdict, why, conf = "REJECTED", "No credible match in licensed list after rewrites.", round(1 - score / 100, 2)

    if use_llm and best and verdict != "REJECTED":
        v = llm_verify(name, address, best)
        log("verifier", result=v)
        if v is None: why += " (Verifier unavailable.)"
        elif verdict == "APPROVED" and not v.get("same_entity"):
            verdict, why = "NEEDS HUMAN REVIEW", f"Verifier rejected approval: {v.get('reason')}"
        elif verdict == "NEEDS HUMAN REVIEW" and v.get("same_entity") and best["consistent"] is True and score >= 80:
            verdict, why = "APPROVED", f"Verifier confirmed: {v.get('reason')}"
    out = {"verdict": verdict, "license_no": best["license_no"] if best and verdict != "REJECTED" else None,
           "confidence": conf, "reasoning": why, "best_match": best, "candidates": evald, "trace": trace}
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        with open(f"{log_dir}/{datetime.datetime.now():%Y%m%d_%H%M%S_%f}.json", "w") as f:
            json.dump({"input": [name, address], **out}, f, default=str, indent=1)
    return out
