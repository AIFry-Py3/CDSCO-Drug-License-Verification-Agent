import os
os.environ["EMBED_BACKEND"] = "fake"   # offline embeddings for tests
import pytest, rag, agent
from test_backend import DF

@pytest.fixture(scope="module")
def idx(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("chroma")); rag.build_index(DF, path=p); return p

def go(idx, n, a=""):
    s = agent.run(n, a, df=DF, use_llm=False, index_path=idx)
    return s, [t["step"] for t in s["trace"]]

def test_rag_retrieves(idx): assert 1 in rag.search("Cipla Ltd Pune", k=3, path=idx)
def test_exact_skips_rag(idx):
    s, steps = go(idx, "Cipla Ltd", "Kurkumbh, Pune 413802")
    assert s["verdict"] == "APPROVED" and s["license_no"] == "M/202" and "rag_tool" not in steps
def test_typo_not_rejected(idx):  # fuzzy alone is decisive here, so RAG is skipped
    s, steps = go(idx, "Sun Pharmaceutcal Industris Ltd", "Baska Vadodara 391450")
    assert s["verdict"] in ("APPROVED", "NEEDS HUMAN REVIEW")
def test_weak_fuzzy_uses_rag(idx):  # shortened name: fuzzy is weak, agent falls back to RAG
    s, steps = go(idx, "Sun Pharma Inds", "Baska Vadodara 391450")
    assert "rag_tool" in steps and s["verdict"] != "REJECTED"
def test_unlisted_triggers_rewrites_then_rejects(idx):
    s, steps = go(idx, "Quantum Biologics Pvt Ltd", "Mumbai 400001")
    assert s["verdict"] == "REJECTED" and steps.count("rewrite") >= 2
def test_near_miss_not_approved(idx):
    assert go(idx, "Cipla Ltd", "Andheri, Mumbai 400093")[0]["verdict"] != "APPROVED"
def test_name_only_not_approved(idx): assert go(idx, "Cipla Ltd")[0]["verdict"] != "APPROVED"
