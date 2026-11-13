"""Parse CDSCO .docx -> data/records.csv.  Usage: python ingest.py path/to/file.docx"""
import re, sys, pandas as pd
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

# header keywords -> field. EDIT if your headers differ.
KEYS = {"name": ["name", "firm", "manufacturer", "company"],
        "address": ["address", "premises", "location"],
        "license_no": ["licen", "lic."]}

STATES = ["Andaman and Nicobar", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chandigarh",
          "Chhattisgarh", "Dadra and Nagar Haveli", "Daman and Diu", "Delhi", "Goa", "Gujarat", "Haryana",
          "Himachal Pradesh", "Jammu & Kashmir", "Jharkhand", "Karnataka", "Kerala", "Lakshadweep",
          "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha",
          "Puducherry", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
          "Uttar Pradesh", "Uttarakhand", "West Bengal"]
_key = lambda t: re.sub(r"[^a-z]", "", t.lower())
ALIASES = {"tamilnadu": "Tamil Nadu", "karanataka": "Karnataka", "orissa": "Odisha",
           "pondicherry": "Puducherry", "chandigarhadmn": "Chandigarh", "jammuandkashmir": "Jammu & Kashmir"}
STATE_KEYS = sorted([(_key(x), x) for x in STATES] + list(ALIASES.items()), key=lambda kv: -len(kv[0]))

def find_state(text):
    k = _key(text)
    return next((name for key, name in STATE_KEYS if key in k), None)

def find_category(text):
    t = text.lower()
    return "Loan" if "loan" in t else "COPP" if "copp" in t else "Original" if "original" in t else None

def blocks(doc):
    for ch in doc.element.body.iterchildren():
        if ch.tag.endswith("}p"): yield Paragraph(ch, doc)
        elif ch.tag.endswith("}tbl"): yield Table(ch, doc)

def map_header(cells):
    m = {}
    for i, c in enumerate(cells):
        for field, kws in KEYS.items():
            if field not in m and any(k in c.lower() for k in kws): m[field] = i
    return m if len(m) == 3 else None

def main(path):
    if path.lower().endswith(".doc"):
        sys.exit("Old .doc format: open it in Word, Save As .docx, then rerun.")
    doc, rows, bad, state, cat, mapping = Document(path), [], [], "", "", None
    for b in blocks(doc):
        if isinstance(b, Paragraph):
            t = b.text.strip()
            if t and len(t) <= 90 and not t.lower().startswith("no. of"):
                st, ct = find_state(t), find_category(t)
                if st: state, cat = st, ct or ""
                elif ct: cat = ct
            continue
        for r in b.rows:
            cells = [c.text.strip() for c in r.cells]
            h = map_header(cells)
            if h: mapping = h; continue          # header row
            if not mapping: bad.append((state, cells)); continue
            try:
                rec = {f: cells[i] for f, i in mapping.items()}
                if len(rec["name"]) < 3 or rec["name"].isdigit(): raise ValueError("bad name")
                rows.append({**rec, "state": state, "category": cat})
            except Exception:
                bad.append((state, cells))
    df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    df.to_csv("data/records.csv", index=False)
    pd.DataFrame(bad, columns=["state", "cells"]).to_csv("data/parse_errors.csv", index=False)
    print(df.groupby(["state", "category"]).size().to_string())
    print(f"{len(df)} records, {len(bad)} unparsed rows (data/parse_errors.csv)")
    if mapping is None: print("NO HEADER MATCHED - edit KEYS in ingest.py")

if __name__ == "__main__": main(sys.argv[1])
