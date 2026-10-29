import json, sys, pathlib, urllib.request
from retrieve import retrieve

ROOT = pathlib.Path(__file__).resolve().parent.parent
URL = "http://127.0.0.1:8080/v1/chat/completions"
REFUSAL = "I cannot find this in the provided documents."
SYSTEM = (
    "You are a careful assistant answering questions about model quantization documentation. "
    "Answer using ONLY the provided context. Do not use outside knowledge. "
    f"If the context does not contain the answer, reply exactly: {REFUSAL} "
    "Answer in English, in at most 4 sentences."
)

_chunks = {r["id"]: r for r in map(json.loads, open(ROOT / "chunks.jsonl", encoding="utf-8"))}

def build_messages(question, chunks):
    ctx = "\n\n".join(f"[{i}] {c['heading']}\n{c['text']}" for i, c in enumerate(chunks, 1))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n\n{ctx}\n\nQuestion: {question}"}]

def generate(messages, url=URL):
    body = json.dumps({"messages": messages, "temperature": 0, "seed": 42,
                       "max_tokens": 256, "cache_prompt": False}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    return {"answer": ch["message"]["content"].strip(), "finish": ch["finish_reason"],
            "usage": d["usage"], "timings": d["timings"], "model": d.get("model", "")}

def run_one(q, mode="retrieved", k=4):
    chunks = ([_chunks[i] for i in q["gold_chunks"]] if mode == "oracle"
              else retrieve(q["question"], k=k))
    out = generate(build_messages(q["question"], chunks))
    out.update(qid=q["qid"], mode=mode, context_ids=[c["id"] for c in chunks])
    return out

if __name__ == "__main__":
    mode = "oracle" if "--oracle" in sys.argv else "retrieved"
    want = [a for a in sys.argv[1:] if not a.startswith("--")]
    qs = {q["qid"]: q for q in map(json.loads, open(ROOT / "eval/eval_40.jsonl", encoding="utf-8"))}
    for qid in want:
        q = qs[qid]
        r = run_one(q, mode)
        print("=" * 80)
        print(f"{qid} [{q['type']}] mode={mode}  context={r['context_ids']}")
        print(f"prompt_tokens={r['usage']['prompt_tokens']}  gen_tokens={r['usage']['completion_tokens']}  finish={r['finish']}")
        print(f"Q: {q['question']}")
        print(f"A: {r['answer']}")
        print(f"GOLD: {q['gold_answer']}")
