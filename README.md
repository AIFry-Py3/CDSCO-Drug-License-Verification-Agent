# CDSCO Licence Verification Agent

Type a manufacturer name and address. The agent checks it against the CDSCO licensed-manufacturer list and returns
**APPROVED / REJECTED / NEEDS HUMAN REVIEW**, the licence number, a confidence score and a visible reasoning trace.

## Run it (5 steps)

1. **Setup** (Python 3.10+; PowerShell shown, use `source .venv/bin/activate` on Mac/Linux)
   ```
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Config:** `copy .env.example .env`, then paste your `GEMINI_API_KEY` (free key: aistudio.google.com/apikey).
   No key? Leave it blank: the agent still runs, just without the Gemini verifier.
3. **Parse the data** (must be `.docx`; if you have `.doc`, open in Word and Save As `.docx`)
   ```
   python ingest.py ../CDSCO_Drug_License_holder_data.docx
   ```
   Writes `data/records.csv` (name, address, license_no, state, category) and `data/parse_errors.csv`.
4. **Build the vector index** (once)
   ```
   python rag.py
   ```
   If Gemini embeddings are busy or rate-limited, set `EMBED_BACKEND=fake` in `.env` and rerun. It uses an offline
   character-trigram embedder. Use the same setting for building and searching.
5. **Launch the app**
   ```
   streamlit run app.py
   ```
   Enter a name and address (include the 6-digit PIN for best accuracy) and click Verify. Each agent step appears live.

Run tests any time: `pytest -q` (32 tests, fully offline; the planner tests use a scripted fake LLM).

## Evaluation, feedback, OCR (v2)
Setup:  
python ingest.py <docx> && python rag.py
pytest -q
python evaluate.py --n 12 --sweep --feedback-demo
streamlit run app.py
```
python evaluate.py --n 12                    # builds data/eval_cases.csv (4 kinds x 12), compares fuzzy-only / RAG-only / agent
python evaluate.py --sweep                   # grid over ambiguous/approve thresholds, best (lowest false-approval) first
python evaluate.py --feedback-demo           # same cases before/after simulated human resolutions
python evaluate.py --llm                     # include Gemini verifier (slower, uses quota)
```
Writes `eval_report.md`. Edit/add rows in `data/eval_cases.csv`; it is reused on later runs. Delete it to regenerate.
- **Feedback:** in the app, NEEDS HUMAN REVIEW results show Approve / Not-listed buttons. Saved to `data/feedback.json`, checked first on every run.
- **Photo:** `Photo` tab. EasyOCR by default (`OCR_ENGINE=tesseract` to use Tesseract). Low confidence triggers a preprocessing retry, then asks for a clearer image.
- **Validity:** if `records.csv` has a `valid_upto` column the agent warns when it is in the past. Otherwise every result carries a note that a listing does not prove current validity.

## How the agent decides (agent.py, LangGraph)

`normalize -> [ planner -> tool ]* -> finalize`, where the tool is one of: fuzzy search, RAG search, rewrite query, decide, verify.

- **The LLM is the planner.** After every step Gemini sees the query, what has been done, the top candidates and the actions
  currently allowed, and picks the next one with a one-sentence reason (shown live in the app and saved in `logs/`).
- **Guardrails.** It can only pick allowed actions (e.g. no `finish` before `decide`), it gets at most `MAX_STEPS` (8) choices,
  and it never gives the verdict: the deterministic `decide` tool does, so the LLM cannot approve a wrong-city match.
- **Fallback.** No API key, a Gemini error, or an invalid choice: the rule-based policy (`rules_next`, the original fixed graph)
  takes over for that step. The trace marks each step `by: llm` or `by: rules`.
- Same name but a different city/PIN is never approved. Name-only input is never approved.
- The verifier can overturn an approval and send the agent back for another attempt.
- Human decisions from the app are checked first on every run (feedback loop).

Compare modes: `python evaluate.py --llm` adds an `agent_llm_planner` row next to the rule-based `agent` row.
Run the planner offline with `planner="rules"` (default when no key or "Use Gemini" is off).
