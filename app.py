import os
import streamlit as st
import agent, rag, feedback
from backend import load

st.set_page_config(page_title="CDSCO Licence Verifier", layout="wide")
st.title("CDSCO Licence Verifier (agent)")
if not os.path.exists(rag.PATH): st.warning("Vector index missing. Run: python rag.py")

@st.cache_data
def get_df(): return load()

tab_t, tab_i = st.tabs(["Type", "Photo"])
with tab_t:
    name = st.text_input("Manufacturer name")
    addr = st.text_area("Address (include city and 6-digit PIN)")
    go_text = st.button("Verify", type="primary")
with tab_i:
    img = st.file_uploader("Label / document photo", type=["png", "jpg", "jpeg"])
    go_img = st.button("Verify photo")
llm = st.checkbox("Use Gemini (LLM planner + verifier)", value=True, help="Off = fully rule-based, no API calls")

def go(**kw):
    final = {}
    with st.status("Agent working...", expanded=True) as status:
        for node, upd in agent.stream(df=get_df(), use_llm=llm, **kw):
            final.update(upd)
            last = upd["trace"][-1] if upd.get("trace") else {}
            if node == "planner":
                st.write(f"**planner ({last['by']})** -> `{last['action']}`" + (f" [{last['strategy']}]" if last.get("strategy") else ""),
                         f"_{last['reason']}_")
            else: st.write(f"**{node}**", {k: v for k, v in last.items() if k != "step"})
        status.update(label="Done", state="complete")
    final.setdefault("name", kw.get("name", "")); final.setdefault("address", kw.get("address", ""))
    st.session_state["final"] = final

if go_text and name.strip(): go(name=name, address=addr)
if go_img and img: go(image=img.getvalue())

f = st.session_state.get("final")
if f:
    {"APPROVED": st.success, "REJECTED": st.error}.get(f["verdict"], st.warning)(f["verdict"])
    st.write(f"**Licence no:** {f['license_no']}  |  **Confidence:** {f['confidence']}")
    st.write(f["reasoning"])
    st.caption("A listing is not proof the licence is currently valid.")
    if f.get("ocr_text"): st.text_area("OCR text read from image", f["ocr_text"], disabled=True)
    if f.get("best"): st.json(f["best"])
    with st.expander("Candidates"): st.dataframe(f.get("cands", []))
    if f["verdict"] == "NEEDS HUMAN REVIEW" and f.get("name"):
        st.subheader("Human review")
        opts = {f"{c['name']} | {c['license_no']}": c for c in f.get("cands", [])}
        pick = st.selectbox("Correct record (if any)", list(opts)) if opts else None
        c1, c2 = st.columns(2)
        if c1.button("Same entity: APPROVE") and pick:
            feedback.save(f["name"], f["address"], "APPROVED", opts[pick]); st.success("Saved. Repeat runs use this.")
        if c2.button("Not listed: REJECT"):
            feedback.save(f["name"], f["address"], "REJECTED"); st.success("Saved. Repeat runs use this.")
