import pandas as pd, pytest
from backend import norm, pin, prep, verify

DF = prep(pd.DataFrame([
    {"name": "Sun Pharmaceutical Industries Limited", "address": "Plot 5, Baska, Vadodara, Gujarat 391450", "license_no": "G/101", "state": "Gujarat"},
    {"name": "Cipla Ltd", "address": "Kurkumbh, Pune, Maharashtra 413802", "license_no": "M/202", "state": "Maharashtra"},
    {"name": "Zenith Labs Pvt Ltd", "address": "Sector 4, Panchkula, Haryana 134109", "license_no": "H/303", "state": "Haryana"},
]))
run = lambda n, a="": verify(n, a, df=DF, use_llm=False, log_dir=None)

def test_norm(): assert norm("Cipla Pvt. Ltd.") == "cipla private limited"
def test_pin(): assert pin("Pune 413802") == "413802" and pin("no pin") == ""
def test_exact_approved():
    r = run("Cipla Ltd", "Kurkumbh, Pune 413802"); assert r["verdict"] == "APPROVED" and r["license_no"] == "M/202"
def test_abbrev_variant_approved():
    assert run("Cipla Limited", "Pune, Maharashtra 413802")["verdict"] == "APPROVED"
def test_typo_not_rejected():
    assert run("Sun Pharmaceutcal Industris Ltd", "Baska Vadodara 391450")["verdict"] in ("APPROVED", "NEEDS HUMAN REVIEW")
def test_unlisted_rejected():
    assert run("Quantum Biologics Pvt Ltd", "Mumbai 400001")["verdict"] == "REJECTED"
def test_near_miss_never_approved():  # same name, different city/PIN
    r = run("Cipla Ltd", "Andheri, Mumbai 400093"); assert r["verdict"] != "APPROVED"
def test_name_only_not_approved():
    assert run("Cipla Ltd")["verdict"] != "APPROVED"
