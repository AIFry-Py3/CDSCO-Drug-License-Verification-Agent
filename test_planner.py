"""Planner tests with a scripted fake LLM (no API key needed): the planner really controls the route,
and the guardrails (valid actions only, step cap, deterministic verdict) hold."""
import os
os.environ["EMBED_BACKEND"] = "fake"
import pytest, rag, agent
from test_backend import DF


@pytest.fixture(scope="module")
def idx(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("chroma")); rag.build_index(DF, path=p); return p


def scripted(*decisions):
    """Fake LLM that replays decisions in order, then returns None (= unavailable)."""
    it = iter(decisions)
    return lambda prompt, allowed, strategies: next(it, None)


def go(idx, planner, n, a=""):
    s = agent.run(n, a, df=DF, use_llm=False, index_path=idx, planner=planner)
    return s, [t["step"] for t in s["trace"]], [t for t in s["trace"] if t["step"] == "planner"]


def act(a, **kw): return {"action": a, "reason": "test", **kw}


def test_llm_choice_overrides_rules(idx):
    # Rules would skip RAG on an exact match; the fake LLM asks for it, and the agent obeys.
    s, steps, plans = go(idx, scripted(act("fuzzy_search"), act("rag_search"), act("decide"), act("finish")),
                         "Cipla Ltd", "Kurkumbh, Pune 413802")
    assert "rag_tool" in steps and s["verdict"] == "APPROVED"
    assert [p["by"] for p in plans] == ["llm"] * 4 and plans[1]["action"] == "rag_search"


def test_rules_mode_unchanged(idx):
    s, steps, plans = go(idx, None, "Cipla Ltd", "Kurkumbh, Pune 413802")
    assert "rag_tool" not in steps and all(p["by"] == "rules" for p in plans)


def test_invalid_action_falls_back_to_rules(idx):
    s, steps, plans = go(idx, scripted(act("launch_missiles")), "Cipla Ltd", "Kurkumbh, Pune 413802")
    assert plans[0]["by"] == "rules" and s["verdict"] == "APPROVED"


def test_cannot_finish_before_deciding(idx):
    s, steps, plans = go(idx, scripted(act("finish")), "Cipla Ltd", "Kurkumbh, Pune 413802")
    assert plans[0]["by"] == "rules" and plans[0]["action"] == "fuzzy_search" and "decide" in steps


def test_llm_picks_rewrite_strategy(idx):
    s, steps, plans = go(idx, scripted(act("fuzzy_search"), act("rewrite_query", strategy="name + address")),
                         "Quantum Biologics Pvt Ltd", "Mumbai 400001")
    rw = [t for t in s["trace"] if t["step"] == "rewrite"]
    assert rw[0]["plan"] == "name + address"


def test_step_cap_terminates(idx):
    always = lambda prompt, allowed, strategies: {"action": "fuzzy_search", "reason": "loop"}
    s, steps, plans = go(idx, always, "Quantum Biologics Pvt Ltd", "Mumbai 400001")
    assert s["verdict"] and steps[-1] == "final"
    assert sum(p["by"] == "llm" for p in plans) == agent.MAX_STEPS   # LLM gets exactly the cap, then rules finish the job
    assert any("Step cap" in p["reason"] for p in plans)


def test_llm_cannot_force_approval_of_near_miss(idx):
    # The LLM rushes to decide/finish; the deterministic decide tool still refuses (wrong city/PIN).
    s, *_ = go(idx, scripted(act("fuzzy_search"), act("decide"), act("finish")), "Cipla Ltd", "Andheri, Mumbai 400093")
    assert s["verdict"] != "APPROVED"


def test_no_key_falls_back(idx, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    s, steps, plans = go(idx, agent.llm_plan, "Cipla Ltd", "Kurkumbh, Pune 413802")
    assert s["verdict"] == "APPROVED" and all(p["by"] == "rules" for p in plans)
