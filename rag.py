"""Vector store (Chroma + Gemini embeddings). Build once: python rag.py"""
import os, time, hashlib, math
import chromadb
from dotenv import load_dotenv
load_dotenv()

PATH, COLL, DIM = "data/chroma", "licenses", 768
EMB_MODEL = "gemini-embedding-001"

def _retry(fn, n=4):
    for i in range(n):
        try: return fn()
        except Exception:
            if i == n - 1: raise
            time.sleep(2 ** i)

def _fake(t, d=256):  # offline embedder for tests only (EMBED_BACKEND=fake)
    v, t = [0.0] * d, f"  {t.lower()} "
    for i in range(len(t) - 2):
        v[int(hashlib.md5(t[i:i + 3].encode()).hexdigest(), 16) % d] += 1
    n = math.sqrt(sum(x * x for x in v)) or 1
    return [x / n for x in v]

def embed(texts, doc=True):
    if os.getenv("EMBED_BACKEND") == "fake": return [_fake(t) for t in texts]
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    cfg = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT" if doc else "RETRIEVAL_QUERY",
                                   output_dimensionality=DIM)
    r = _retry(lambda: client.models.embed_content(model=os.getenv("EMBED_MODEL", EMB_MODEL),
                                                   contents=texts, config=cfg))
    return [e.values for e in r.embeddings]

def build_index(df, path=PATH, batch=100):
    client = chromadb.PersistentClient(path=path)
    try: client.delete_collection(COLL)
    except Exception: pass
    col = client.create_collection(COLL, metadata={"hnsw:space": "cosine"})
    docs = (df["name"] + " | " + df["address"]).tolist()
    for s in range(0, len(df), batch):
        e = min(s + batch, len(df))
        col.add(ids=[str(i) for i in range(s, e)], embeddings=embed(docs[s:e]), documents=docs[s:e],
                metadatas=[{"license_no": df.loc[i, "license_no"], "state": df.loc[i, "state"]} for i in range(s, e)])
        print(f"indexed {e}/{len(df)}"); time.sleep(0.5)

def search(query, k=5, path=PATH):
    col = chromadb.PersistentClient(path=path).get_collection(COLL)
    r = col.query(query_embeddings=embed([query], doc=False), n_results=k)
    return [int(i) for i in r["ids"][0]]   # ids = row positions in records.csv

if __name__ == "__main__":
    from backend import load
    build_index(load())
