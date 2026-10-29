import json, pathlib, sys
import numpy as np, faiss
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_rows = [json.loads(l) for l in open(ROOT / "chunks.jsonl", encoding="utf-8")]
_index = faiss.read_index(str(ROOT / "index.faiss"))
_model = SentenceTransformer(MODEL)

def retrieve(query, k=4):
    """回傳前 k 個最相關的 chunk（含相似度分數）。"""
    q = _model.encode([QUERY_PREFIX + query], normalize_embeddings=True).astype("float32")
    scores, ids = _index.search(q, k)
    return [{**_rows[i], "score": float(s)} for s, i in zip(scores[0], ids[0])]

if __name__ == "__main__":
    # 截斷檢查：bge 的上限是 512 tokens
    tk = _model.tokenizer
    lens = [len(tk.encode(f"{r['heading']}\n\n{r['text']}")) for r in _rows]
    print(f"bge tokens: max {max(lens)}, >512: {sum(n > 512 for n in lens)} / {len(lens)}\n")

    queries = sys.argv[1:] or [
        "What is the difference between post-training quantization and quantization-aware training?",
        "What is the formula for mapping float values to quantized values in ONNX Runtime?",
        "How much does dynamic range quantization reduce model size in TFLite?",
        "Why is float16 quantization useful for GPU delegates?",
        "How do I run a GGUF Q4_K_M model with llama.cpp?",
    ]
    for q in queries:
        print("=" * 80, f"\nQ: {q}")
        for r in retrieve(q):
            print(f"  {r['score']:.3f}  [{r['id']}]  {r['heading'][:90]}")
