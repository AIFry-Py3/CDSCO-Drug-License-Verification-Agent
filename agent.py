"""LangGraph agent. A planner node (Gemini) picks the next tool each step; rules are the fallback + guardrails.

normalize -> [planner -> fuzzy | rag | rewrite | decide | verifier]* -> finalize
The LLM chooses the PROCESS; the verdict still comes from the deterministic `decide` tool, so it cannot approve on its own.
"""
import os, json, time, datetime
from typing import TypedDict
import pandas as pd
from langgraph.graph import StateGraph, END, START
import backend as B, rag, feedback, ocr

MAX_VERIFIER_RUNS = 2
MAX_STEPS = 8   # planner steps before the rule-based policy takes over

def validity_note(best):
    d = str(best.get("valid_upto") or "").strip()
    d = pd.to_datetime(d, dayfirst=True, errors="coerce") if d else pd.NaT
    if pd.isna(d): return " Note: no validity date in the list, so a listing does not prove the licence is currently valid."
    if d < pd.Timestamp.now(): return f" WARNING: listing shows validity ended {d:%d %b %Y}; confirm current status with the licensing authority."
    return f" Listing shows validity until {d:%d %b %Y}; confirm with the licensing authority."


def llm_plan(prompt, allowed, strategies):
    """Ask Gemini for the next action. Returns a dict, or None (no key / error / unparsable) -> rules take over."""
    if not os.getenv("GEMINI_API_KEY"): return None
    from gem import generate_json
    d = generate_json(prompt)
    return d if isinstance(d, dict) else None


def rules_next(s):
    """Rule-based policy = the original fixed graph. Used as fallback when the LLM is unavailable or invalid."""
    last, c = s.get("last"), s.get("cands", [])
    if last in (None, "normalize", "rewrite"): return "fuzzy_search", None
    if last == "fuzzy":
        decisive = s["attempts"] == 0 and c and c[0]["score"] >= B.APPROVE and c[0]["consistent"] is True
        return ("decide" if decisive else "rag_search"), None
    if last == "rag":
        top = max([x["score"] for x in c], default=0)
        return ("rewrite_query" if top < B.AMBIG and s["plans"] else "decide"), None
    if last == "decide":
        return ("verify" if s["use_llm"] and s["verdict"] != "REJECTED" and s.get("best") else "finish"), None
    retry = (s.get("verifier") and not s["verifier"].get("same_entity") and s["plans"]
             and s["verifier_runs"] < MAX_VERIFIER_RUNS)
    return ("rewrite_query" if retry else "finish"), None


class State(TypedDict, total=False):
    name: str; address: str; use_llm: bool
    qn: str; qa: str; qp: str
    query: str; qname: str; qcol: str; pin_only: bool; via: str; attempts: int; plans: list
    cands: list; best: dict; verdict: str; reasoning: str; confidence: float; license_no: str
    verifier: dict; verifier_runs: int; trace: list
    image: bytes; ocr_text: str; ocr_conf: float; locked: bool
    next_action: str; strategy: str; plan_reason: str; steps: int; decided: bool; last: str; searches: int

def build(df, index_path=rag.PATH, planner_fn=None):
    def log(s, step, **kw): return s.get("trace", []) + [{"step": step, **kw}]
    def show(c): return [(df.loc[x["idx"], "name"], round(x["score"], 1), x["src"]) for x in c]

    def enrich(cands, s):
        for c in cands:
            r = df.loc[c["idx"]]
            c.update(B.compare(r, s["qa"], s["qp"]), name=r["name"], address=r["address"],
                     license_no=r["license_no"], state=r.get("state", ""), valid_upto=r.get("valid_upto", ""))
        return cands

    # ---- nodes ----
    def normalize(s):
        qn, qa, qp = B.core(s["name"]), B.norm(s["address"]), B.pin(s["address"])
        core = " ".join(w for w in qn.split() if w not in B.GENERIC) or qn
        plans = [("drop generic words", core, "name_n", False),
                 ("name + address", f"{core} {qa}".strip(), "full_n", False)]
        if qp: plans.append(("search by PIN", core, "name_n", True))
        plans = [(l, q, c, p, core) for l, q, c, p in plans]   # last item: name-only text used for SCORING
        out = {"qn": qn, "qa": qa, "qp": qp, "query": qn, "qname": qn, "qcol": "name_n", "pin_only": False,
               "via": "direct", "attempts": 0, "verifier_runs": 0, "plans": plans, "cands": [],
               "trace": log(s, "normalize", name=qn, address=qa, pin=qp)}
        fb = feedback.lookup(s["name"], s["address"])   # human-confirmed decisions win
        if fb:
            ok = fb["verdict"] == "APPROVED"
            out.update(locked=True, verdict=fb["verdict"], confidence=1.0,
                       reasoning=f"Confirmed by human review on {fb['at'][:10]}.",
                       best={k: fb.get(k) for k in ("name", "address", "license_no", "valid_upto")} if ok else None,
                       trace=log(out, "feedback_hit", verdict=fb["verdict"]))
        return out

    def ocr_node(s):  # TOOL 0: OCR with preprocessing retry
        try: r = ocr.read(s["image"])
        except Exception as e:
            return {"locked": True, "verdict": "NEEDS HUMAN REVIEW", "confidence": 0.0, "best": None, "cands": [],
                    "reasoning": f"OCR failed ({str(e)[:120]}). Type the name and address instead.",
                    "trace": log(s, "ocr_error", error=str(e)[:200])}
        upd = {"ocr_text": r["text"], "ocr_conf": r["conf"],
               "trace": log(s, "ocr_tool", attempts=r["attempts"], conf=r["conf"], text=r["text"][:300])}
        if r["conf"] < ocr.LOW or not r["text"].strip():
            upd.update(locked=True, verdict="NEEDS HUMAN REVIEW", confidence=r["conf"], best=None, cands=[],
                       reasoning="OCR confidence too low even after preprocessing. Upload a clearer image or type the name and address.")
        return upd

    def extract(s):  # LLM/heuristic query extraction
        q = ocr.extract(s["ocr_text"])
        return {"name": q["name"], "address": q["address"], "trace": log(s, "extract_query", **q)}

    def fuzzy(s):  # TOOL 1: fuzzy match on the pandas table
        sub = df[df["pin"] == s["qp"]] if s["pin_only"] else df
        # retrieve with qcol (may include address) but SCORE on name only, so a real address can't inflate a fake firm
        found = [(i, B.score(s["qname"], df.loc[i, "name_n"])) for i, _ in B.top(sub, s["query"], s["qcol"])]
        cands = enrich([{"idx": i, "score": sc, "src": ["fuzzy"]} for i, sc in sorted(found, key=lambda x: -x[1])], s)
        return {"cands": cands, "searches": s.get("searches", 0) + 1, "decided": False,
                "trace": log(s, "fuzzy_tool", query=s["query"], via=s["via"], top=show(cands))}

    def rag_node(s):  # TOOL 2: semantic retrieval (RAG) from Chroma
        q = s["query"] if s["qcol"] == "full_n" else f"{s['query']} {s['qa']}".strip()
        try: ids = rag.search(q, k=5, path=index_path)
        except Exception as e: return {"searches": s.get("searches", 0) + 1, "decided": False,
                                       "trace": log(s, "rag_error", error=str(e)[:200])}
        if s["pin_only"]: ids = [i for i in ids if df.loc[i, "pin"] == s["qp"]]
        merged, new = {c["idx"]: c for c in s["cands"]}, []
        for i in ids:
            if i in merged: merged[i]["src"].append("rag")
            else:
                merged[i] = {"idx": i, "score": B.score(s["qname"], df.loc[i, "name_n"]), "src": ["rag"]}
                new.append(df.loc[i, "name"])
        cands = enrich(sorted(merged.values(), key=lambda c: -c["score"])[:5], s)
        return {"cands": cands, "searches": s.get("searches", 0) + 1, "decided": False,
                "trace": log(s, "rag_tool", retrieved=[df.loc[i, "name"] for i in ids],
                                             new_vs_fuzzy=new, top=show(cands))}

    def rewrite(s):  # planning/self-correction: next query reformulation
        plans = list(s["plans"])   # the planner may pick which reformulation to try; default is the next one
        label, q, col, pin_only, qname = plans.pop(next((k for k, p in enumerate(plans) if p[0] == s.get("strategy")), 0))
        return {"query": q, "qname": qname, "qcol": col, "pin_only": pin_only, "via": label, "attempts": s["attempts"] + 1,
                "plans": plans, "trace": log(s, "rewrite", plan=label, query=q)}

    def decide(s):  # Matcher role: propose verdict
        c = s["cands"]
        plaus = [x for x in c if x["score"] >= B.AMBIG]
        best = max(plaus, key=lambda e: (e["consistent"] is True, e["score"])) if plaus else (c[0] if c else None)
        score = best["score"] if best else 0
        if best and score >= B.AMBIG:
            if s["via"] == "direct" and score >= B.APPROVE and best["consistent"] is True:
                v, why = "APPROVED", "Strong name match and address/PIN consistent."
            else:
                v = "NEEDS HUMAN REVIEW"
                why = (f"Matched via '{s['via']}'. " if s["via"] != "direct" else "") + \
                      ("Name matches but address/PIN inconsistent or missing." if score >= B.APPROVE else "Name match is ambiguous.")
            conf = round(score / 100, 2)
        else:
            v, why, conf = "REJECTED", "No credible match in licensed list after rewrites.", round(1 - score / 100, 2)
        return {"best": best, "verdict": v, "reasoning": why, "confidence": conf, "decided": True,
                "trace": log(s, "decide", proposed=v, why=why)}

    def verifier(s):  # Verifier role: critique the proposal
        v = B.llm_verify(s["name"], s["address"], s["best"])
        return {"verifier": v, "verifier_runs": s["verifier_runs"] + 1, "trace": log(s, "verifier", result=v)}

    def finalize(s):
        v, why, best, ver = s["verdict"], s["reasoning"], s.get("best"), s.get("verifier")
        if s["use_llm"] and v != "REJECTED" and not s.get("locked"):
            if ver is None: why += " (LLM verifier not run.)"
            elif v == "APPROVED" and not ver.get("same_entity"):
                v, why = "NEEDS HUMAN REVIEW", f"Verifier rejected approval: {ver.get('reason')}"
            elif v == "NEEDS HUMAN REVIEW" and ver.get("same_entity") and best["consistent"] is True and best["score"] >= 80:
                v, why = "APPROVED", f"Verifier confirmed: {ver.get('reason')}"
        if best and v != "REJECTED": why += validity_note(best)
        lic = best["license_no"] if best and v != "REJECTED" else None
        out = {"verdict": v, "reasoning": why, "license_no": lic, "trace": log(s, "final", verdict=v)}
        os.makedirs("logs", exist_ok=True)
        with open(f"logs/{datetime.datetime.now():%Y%m%d_%H%M%S_%f}.json", "w") as f:
            json.dump({k: x for k, x in {**s, **out}.items() if k != "image"}, f, default=str, indent=1)
        return out

    # ---- planner: the LLM picks the next action; guardrails limit what it may pick ----
    def allowed_actions(s):
        a = {"fuzzy_search": "fuzzy-match the current query against the licence table",
             "rag_search": "semantic search (Chroma) with the current query; catches spelling/format variants"}
        if s.get("plans"): a["rewrite_query"] = "switch to another query formulation (give a 'strategy')"
        if s.get("searches", 0) >= 1: a["decide"] = "run the deterministic decision tool on the candidates found so far"
        if s.get("decided"):
            a["finish"] = "stop and return the decided verdict"
            if s["use_llm"] and s["verdict"] != "REJECTED" and s.get("best") and s["verifier_runs"] < MAX_VERIFIER_RUNS:
                a["verify"] = "ask the LLM verifier whether the best candidate is the same legal entity at the same place"
        return a

    def plan_prompt(s, allowed):
        hist = []
        for t in s["trace"][-8:]:
            d = {k: v for k, v in t.items() if k != "step"}
            hist.append(f"- planner chose {t['action']}" if t["step"] == "planner" else f"- {t['step']}: {json.dumps(d, default=str)[:260]}")
        cands = "\n".join(f"- {c['name']} | score {c['score']:.0f} | location consistent: {c['consistent']} | {c.get('state', '')}"
                          for c in s.get("cands", [])[:3]) or "- none yet"
        acts = "\n".join(f"- {k}: {v}" for k, v in allowed.items())
        strat = [p[0] for p in s.get("plans", [])]
        return (f"You plan the next step for an agent that checks whether a drug manufacturer is on the CDSCO licensed list.\n"
                f"Query: name={s['name']!r} address={s['address']!r} pin={s['qp'] or 'none'}\n"
                f"Current search query: {s['query']!r} (formulation: {s['via']})\nSteps so far:\n" + ("\n".join(hist) or "- none") +
                f"\nTop candidates:\n{cands}\nVerdict decided yet: {bool(s.get('decided'))}"
                f"{' (' + s['verdict'] + ')' if s.get('decided') else ''}\nStep {s.get('steps', 0) + 1} of {MAX_STEPS}.\n\n"
                f"Allowed actions now:\n{acts}\nrewrite_query strategies available: {strat}\n\n"
                "Rules: choose only the NEXT action. You never give the verdict; the decide tool does. Be economical: if a fuzzy match is "
                "strong and the location is consistent, go straight to decide. If matches are weak or missing, use rag_search or "
                "rewrite_query (pick the strategy most likely to help). Do not repeat an action with the same query. "
                "Use verify only when the case is ambiguous or borderline.\n"
                'Reply JSON only: {"action": "<one allowed action>", "strategy": "<strategy or null>", "reason": "<one sentence>"}')

    def planner(s):
        steps, allowed = s.get("steps", 0) + 1, allowed_actions(s)
        by, act, strat, why = "rules", None, None, ""
        if steps <= MAX_STEPS and planner_fn:
            d = planner_fn(plan_prompt(s, allowed), list(allowed), [p[0] for p in s.get("plans", [])])
            if isinstance(d, dict) and d.get("action") in allowed:
                by, act, why = "llm", d["action"], str(d.get("reason", ""))[:300]
                if act == "rewrite_query" and d.get("strategy") in [p[0] for p in s["plans"]]: strat = d["strategy"]
            else: why = "LLM unavailable or invalid choice; rule-based fallback."
        elif planner_fn: why = f"Step cap ({MAX_STEPS}) reached; rule-based policy takes over."
        else: why = "Rule-based policy."
        if act is None: act, strat = rules_next(s)
        return {"next_action": act, "strategy": strat, "plan_reason": why, "steps": steps,
                "trace": log(s, "planner", by=by, action=act, strategy=strat, reason=why)}

    def tag(name, f): return lambda s: {**f(s), "last": name}

    # ---- conditional edges ----
    def start(s): return "ocr" if s.get("image") else "normalize"
    def after_ocr(s): return "finalize" if s.get("locked") else "extract"
    def after_normalize(s): return "finalize" if s.get("locked") else "planner"
    ROUTE = {"fuzzy_search": "fuzzy", "rag_search": "rag", "rewrite_query": "rewrite", "decide": "decide",
             "verify": "verifier", "finish": "finalize"}

    g = StateGraph(State)
    for n, f in [("ocr", ocr_node), ("extract", extract), ("normalize", tag("normalize", normalize)), ("planner", planner),
                 ("fuzzy", tag("fuzzy", fuzzy)), ("rag", tag("rag", rag_node)), ("rewrite", tag("rewrite", rewrite)),
                 ("decide", tag("decide", decide)), ("verifier", tag("verifier", verifier)), ("finalize", finalize)]:
        g.add_node(n, f)
    g.add_conditional_edges(START, start, {"ocr": "ocr", "normalize": "normalize"})
    g.add_conditional_edges("ocr", after_ocr, {"finalize": "finalize", "extract": "extract"})
    g.add_edge("extract", "normalize")
    g.add_conditional_edges("normalize", after_normalize, {"finalize": "finalize", "planner": "planner"})
    for n in ("fuzzy", "rag", "rewrite", "decide", "verifier"): g.add_edge(n, "planner")
    g.add_conditional_edges("planner", lambda s: ROUTE[s["next_action"]], {v: v for v in set(ROUTE.values())})
    g.add_edge("finalize", END)
    return g.compile()

def _resolve_planner(planner, use_llm):
    """'auto': Gemini planner when use_llm and a key exist, else rules. 'rules': never use the LLM. 'llm': force it. Or a callable."""
    if callable(planner): return planner
    if planner == "llm" or (planner == "auto" and use_llm and os.getenv("GEMINI_API_KEY")): return llm_plan
    return None

def stream(name="", address="", df=None, use_llm=True, index_path=rag.PATH, image=None, planner="auto"):
    """Yields (node, update) as each node finishes - for live display."""
    df = B.load() if df is None else df
    init = {"name": name, "address": address, "use_llm": use_llm, "trace": [], "cands": []}
    if image: init["image"] = image
    graph = build(df, index_path, _resolve_planner(planner, use_llm))
    for chunk in graph.stream(init, {"recursion_limit": 80}, stream_mode="updates"):
        for node, upd in chunk.items(): yield node, upd

def run(name="", address="", df=None, use_llm=True, index_path=rag.PATH, image=None, planner="auto"):
    state = {}
    for _, upd in stream(name, address, df, use_llm, index_path, image, planner): state.update(upd)
    return state
