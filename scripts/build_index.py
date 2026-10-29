import json, pathlib, time
import numpy as np, faiss
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = "BAAI/bge-small-en-v1.5"

rows = [json.loads(l) for l in open(ROOT / "chunks.jsonl", encoding="utf-8")]
texts = [f"{r['heading']}\n\n{r['text']}" for r in rows]

model = SentenceTransformer(MODEL)
t0 = time.time()
emb = model.encode(texts, batch_size=32, normalize_embeddings=True,
                   show_progress_bar=True, convert_to_numpy=True).astype("float32")
print(f"embedded {emb.shape} in {time.time()-t0:.1f}s")

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, str(ROOT / "index.faiss"))
print("index size:", index.ntotal, "| saved to index.faiss")
