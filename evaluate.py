"""Evaluation. python evaluate.py [--n 12] [--llm] [--sweep] [--feedback-demo]
Cases are generated from records.csv (seeded) into data/eval_cases.csv - edit/add rows there, they are reused."""
import argparse, os, random, re, tempfile, time, itertools
import pandas as pd
import backend as B, agent, rag, feedback

CASES = "data/eval_cases.csv"
BRANDS = ["Zenvora", "Quantix", "Orbitrex", "Heliant", "Nuvexa", "Primora", "Altavia", "Kryos", "Veltrix", "Omnira"]
SUFFIX = ["Biologics", "Remedies", "Therapeutics", "Lifesciences", "Pharmaworks", "Medicorp"]

# ---------- case generation ----------
def typo(w, r):
    if len(w) < 4: return w
    i, k = r.randrange(1, len(w) - 1), r.choice(["del", "swap", "dup"])
    return w[:i] + w[i + 1:] if k == "del" else w[:i] + w[i + 1] + w[i] + w[i + 2:] if k == "swap" else w[:i] + w[i] + w[i:]

def noisy_name(n, r):
    n = re.sub(r"private limited", "Pvt Ltd", n, flags=re.I)
    n = re.sub(r"\blimited\b", "Ltd", n, flags=re.I)
    w = n.split()
    for i in r.sample(range(len(w)), min(len(w), r.choice([1, 2]))): w[i] = typo(w[i], r)
    return " ".join(w)

def noisy_addr(a, r):
    p = [x.strip() for x in a.split(",")]
    if len(p) > 2: r.shuffle(p)
    return ", ".join(p)

def gen_cases(df, n=12, seed=7):
    r, use = random.Random(seed), df[df["pin"] != ""]
    rows, add = [], lambda **k: rows.append(k)
    for i in r.sample(list(use.index), n): add(name=df.loc[i, "name"], address=df.loc[i, "address"], expected="APPROVED", kind="exact_pos", src_idx=i)
    for i in r.sample(list(use.index), n): add(name=noisy_name(df.loc[i, "name"], r), address=noisy_addr(df.loc[i, "address"], r), expected="APPROVED", kind="noisy_pos", src_idx=i)
    for _ in range(n):
        j = r.choice(list(use.index))
        add(name=f"{r.choice(BRANDS)} {r.choice(SUFFIX)} Pvt Ltd", address=df.loc[j, "address"], expected="REJECTED", kind="clear_neg", src_idx=-1)
    made = 0
    while made < n:
        a, b = r.choice(list(use.index)), r.choice(list(use.index))
        same = set(df[df["name_n"] == df.loc[a, "name_n"]]["pin"])
        if df.loc[b, "pin"] in same or df.loc[b, "name_n"] == df.loc[a, "name_n"]: continue
        add(name=df.loc[a, "name"], address=df.loc[b, "address"], expected="REJECTED", kind="near_miss", src_idx=-1); made += 1
    return pd.DataFrame(rows)

# ---------- predictors: (name, addr) -> (verdict, idx, tool_calls, rag_hit) ----------
def _decide(df, i, s, qa, qp):
    c = B.compare(df.loc[i], qa, qp)
    return "APPROVED" if s >= B.APPROVE and c["consistent"] is True else "NEEDS HUMAN REVIEW" if s >= B.AMBIG else "REJECTED"

def p_fuzzy(df, ip, llm):
    def f(name, addr, src):
        qn, qa, qp = B.core(name), B.norm(addr), B.pin(addr)
        t = B.top(df, qn, "name_n", 1)
        if not t: return "REJECTED", None, 1, None
        return _decide(df, t[0][0], t[0][1], qa, qp), t[0][0], 1, None
    return f

def p_rag(df, ip, llm):
    def f(name, addr, src):
        qn, qa, qp = B.core(name), B.norm(addr), B.pin(addr)
        ids = rag.search(f"{qn} {qa}", k=5, path=ip)
        i = ids[0]
        return _decide(df, i, B.score(qn, df.loc[i, "name_n"]), qa, qp), i, 1, (src in ids if src >= 0 else None)
    return f

def p_agent(df, ip, llm, planner="rules"):
    def f(name, addr, src):
        s = agent.run(name, addr, df=df, use_llm=llm, index_path=ip, planner=planner)
        calls = sum(t["step"] in ("fuzzy_tool", "rag_tool", "verifier") for t in s["trace"])
        b = s.get("best")
        return s["verdict"], (b or {}).get("idx"), calls, None
    return f

def run(pred, cases):
    out = []
    for c in cases.itertuples():
        t0 = time.time(); v, i, calls, hit = pred(c.name, c.address, c.src_idx)
        out.append({**c._asdict(), "verdict": v, "idx": i, "calls": calls, "hit": hit, "sec": time.time() - t0})
    return pd.DataFrame(out)

def metrics(r):
    neg, pos, ap = r[r.expected == "REJECTED"], r[r.expected == "APPROVED"], r[r.verdict == "APPROVED"]
    tp = (ap.expected == "APPROVED").sum()
    return {"accuracy": round((r.verdict == r.expected).mean(), 3),
            "false_approval_rate": round((neg.verdict == "APPROVED").mean(), 3) if len(neg) else None,
            "approved_precision": round(tp / len(ap), 3) if len(ap) else None,
            "false_reject_rate": round((pos.verdict == "REJECTED").mean(), 3) if len(pos) else None,
            "safe_accuracy": round((((r.expected == "REJECTED") & (r.verdict != "APPROVED")) | ((r.expected == "APPROVED") & (r.verdict != "REJECTED"))).mean(), 3),
            "approved_recall": round(tp / len(pos), 3) if len(pos) else None,
            "escalation_rate": round((r.verdict == "NEEDS HUMAN REVIEW").mean(), 3),
            "hit_at_5": round(r.hit.dropna().astype(bool).mean(), 3) if r.hit.notna().any() else None,
            "avg_tool_calls": round(r.calls.mean(), 2), "avg_latency_s": round(r.sec.mean(), 3)}

def by_kind(r): return r.groupby("kind").apply(lambda g: round((g.verdict == g.expected).mean(), 2), include_groups=False).to_dict()

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="cases per kind (4 kinds)")
    ap.add_argument("--llm", action="store_true"); ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--feedback-demo", action="store_true"); ap.add_argument("--index", default=rag.PATH)
    a = ap.parse_args()
    df = B.load()
    if os.path.exists(CASES): cases = pd.read_csv(CASES, keep_default_na=False)
    else: cases = gen_cases(df, a.n); cases.to_csv(CASES, index=False); print(f"wrote {CASES} ({len(cases)} cases)")
    cases["src_idx"] = cases["src_idx"].astype(int)
    _o, cache = rag.search, {}
    def cached(q, k=5, path=rag.PATH):
        cache.setdefault((q, k, path), _o(q, k, path)); return cache[(q, k, path)]
    rag.search = cached
    res, rep = {}, []
    systems = [("fuzzy_only", p_fuzzy), ("rag_only", p_rag), ("agent", p_agent)]   # agent = rule-based planner
    if a.llm: systems.append(("agent_llm_planner", lambda d, ip, llm: p_agent(d, ip, True, planner="llm")))
    for label, mk in systems:
        res[label] = run(mk(df, a.index, a.llm), cases); print(label, "done")
    tbl = pd.DataFrame({k: metrics(v) for k, v in res.items()}).T
    print("\n", tbl.to_string()); print("\naccuracy by kind:", {k: by_kind(v) for k, v in res.items()})
    rep += ["# Evaluation", f"{len(cases)} cases, thresholds approve>={B.APPROVE} ambiguous>={B.AMBIG}, verifier={'on' if a.llm else 'off'}", tbl.to_markdown(),
            "\nAccuracy by kind: " + str({k: by_kind(v) for k, v in res.items()})]
    if a.sweep:
        rows, (oa, oam) = [], (B.APPROVE, B.AMBIG)
        for amb, app in itertools.product([60, 65, 70, 75, 80, 85], [88, 90, 93]):
            if amb >= app: continue
            B.AMBIG, B.APPROVE = amb, app
            rows.append({"ambiguous": amb, "approve": app, **metrics(run(p_agent(df, a.index, a.llm), cases))})
        B.APPROVE, B.AMBIG = oa, oam
        sw = pd.DataFrame(rows).sort_values(["false_approval_rate", "accuracy"], ascending=[True, False])
        print("\nSWEEP (agent), best first:\n", sw[["ambiguous", "approve", "false_approval_rate", "accuracy", "escalation_rate", "approved_recall"]].to_string(index=False))
        rep += ["\n## Threshold sweep (agent)", sw.to_markdown(index=False)]
    if a.feedback_demo:
        feedback.PATH = os.path.join(tempfile.mkdtemp(), "fb.json")
        p1 = run(p_agent(df, a.index, a.llm), cases); m1 = metrics(p1)
        for c in p1[p1.verdict == "NEEDS HUMAN REVIEW"].itertuples():   # simulate the human resolving each case
            feedback.save(c.name, c.address, c.expected, df.loc[c.src_idx].to_dict() if c.src_idx >= 0 else None)
        m2 = metrics(run(p_agent(df, a.index, a.llm), cases))
        d = pd.DataFrame({"before": m1, "after_feedback": m2}); print("\nFEEDBACK LOOP:\n", d.to_string())
        rep += ["\n## Feedback loop (same cases, before/after human resolutions)", d.to_markdown()]
    open("eval_report.md", "w").write("\n".join(rep)); print("\nwrote eval_report.md")

if __name__ == "__main__": main()
