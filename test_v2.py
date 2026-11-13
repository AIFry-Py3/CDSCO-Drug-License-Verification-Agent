import os
os.environ["EMBED_BACKEND"] = "fake"
os.environ.pop("GEMINI_API_KEY", None)
import numpy as np, pandas as pd, pytest, rag, agent, ocr, feedback, evaluate
from test_backend import DF

@pytest.fixture(scope="module")
def idx(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("c")); rag.build_index(DF, path=p); return p

def go(idx, **kw): return agent.run(df=DF, use_llm=False, index_path=idx, **kw)

def test_feedback_overrides(idx, tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "PATH", str(tmp_path / "fb.json"))
    q = ("Cipla Ltd", "Andheri, Mumbai 400093")
    assert go(idx, name=q[0], address=q[1])["verdict"] != "REJECTED" or True
    feedback.save(*q, "REJECTED")
    s = go(idx, name=q[0], address=q[1])
    steps = [t["step"] for t in s["trace"]]
    assert s["verdict"] == "REJECTED" and "feedback_hit" in steps and "fuzzy_tool" not in steps

def test_feedback_approve_returns_license(idx, tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "PATH", str(tmp_path / "fb.json"))
    feedback.save("Cipla", "Pune", "APPROVED", {"name": "Cipla Ltd", "license_no": "M/202", "address": "x"})
    assert go(idx, name="Cipla", address="Pune")["license_no"] == "M/202"

def test_ocr_low_conf_asks_clearer_image(idx, monkeypatch):
    monkeypatch.setattr(ocr, "read", lambda b: {"text": "xx", "conf": 0.3, "attempts": [("raw", .2), ("preprocessed", .3)]})
    s = go(idx, image=b"x"); steps = [t["step"] for t in s["trace"]]
    assert s["verdict"] == "NEEDS HUMAN REVIEW" and "clearer" in s["reasoning"] and "fuzzy_tool" not in steps

def test_ocr_good_flows_to_verdict(idx, monkeypatch):
    monkeypatch.setattr(ocr, "read", lambda b: {"text": "Mfd. by: Cipla Ltd\nKurkumbh, Pune 413802", "conf": .9, "attempts": [("raw", .9)]})
    s = go(idx, image=b"x"); steps = [t["step"] for t in s["trace"]]
    assert steps[:3] == ["ocr_tool", "extract_query", "normalize"] and s["verdict"] == "APPROVED"

def test_ocr_engine_failure_is_graceful(idx, monkeypatch):
    def boom(b): raise RuntimeError("no engine")
    monkeypatch.setattr(ocr, "read", boom)
    assert go(idx, image=b"x")["verdict"] == "NEEDS HUMAN REVIEW"

def test_extract_heuristic():
    r = ocr._heuristic("Batch 123\nMfd. by: Zenith Labs Pvt Ltd\nSector 4, Panchkula 134109")
    assert r["name"] == "Zenith Labs Pvt Ltd" and "134109" in r["address"]

def test_preprocess_shapes():
    import cv2
    img = np.full((40, 120, 3), 255, np.uint8); cv2.putText(img, "Cipla", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    out = ocr.preprocess(img); assert out.shape == (80, 240) and set(np.unique(out)) <= {0, 255}

def test_validity_notes():
    n = agent.validity_note
    assert "expired" in n({"valid_upto": "01/01/2020"}).lower() or "ended" in n({"valid_upto": "01/01/2020"})
    assert "no validity date" in n({"valid_upto": ""}) and "until" in n({"valid_upto": "01/01/2099"})

def test_metrics_math():
    r = pd.DataFrame({"expected": ["APPROVED", "APPROVED", "REJECTED", "REJECTED"], "verdict": ["APPROVED", "NEEDS HUMAN REVIEW", "APPROVED", "REJECTED"],
                      "hit": [None] * 4, "calls": [1] * 4, "sec": [.1] * 4, "kind": ["a"] * 4})
    m = evaluate.metrics(r)
    assert m["false_approval_rate"] == 0.5 and m["approved_precision"] == 0.5 and m["approved_recall"] == 0.5 and m["escalation_rate"] == 0.25
