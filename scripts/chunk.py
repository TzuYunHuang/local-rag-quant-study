import json, re, pathlib, statistics, collections
from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
OUT = ROOT / "chunks.jsonl"
TARGET, OVERLAP = 400, 60

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
btok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
def ntok(s): return max(len(tok.encode(s, add_special_tokens=False)),
                        len(btok.encode(s, add_special_tokens=False)))

def read_doc(path):
    text = path.read_text(encoding="utf-8")
    m = re.match(r"<!-- source: (.*?) -->\n+", text)
    return (m.group(1) if m else ""), (text[m.end():] if m else text)

def split_sections(text):
    """依標題切成 [(標題路徑, 標題, 內文)]，程式碼區塊內的 # 不算標題。"""
    secs, stack, buf, in_code = [], [], [], False
    def flush():
        body = "\n".join(buf).strip()
        if body and stack:
            secs.append((" > ".join(t for _, t in stack), stack[-1][1], body))
        elif body:
            secs.append(("(intro)", "", body))
        buf.clear()
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if m:
            flush()
            lvl = len(m.group(1))
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, m.group(2)))
        else:
            buf.append(line)
    flush()
    return secs

def blocks(body):
    """依空行切段落，程式碼區塊保持完整。"""
    out, cur, in_code = [], [], False
    for line in body.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        cur.append(line)
        if not in_code and line.strip() == "":
            b = "\n".join(cur).strip()
            if b: out.append(b)
            cur = []
    b = "\n".join(cur).strip()
    if b: out.append(b)
    return out

def hard_split(block):
    parts, cur, n = [], [], 0
    for line in block.split("\n"):
        k = ntok(line) + 1
        if cur and n + k > TARGET:
            parts.append("\n".join(cur)); cur, n = [], 0
        cur.append(line); n += k
    if cur: parts.append("\n".join(cur))
    return parts

def pack(text):
    bl = []
    for b in blocks(text):
        bl += hard_split(b) if ntok(b) > TARGET else [b]
    chunks, cur, n = [], [], 0
    for b in bl:
        k = ntok(b)
        if cur and n + k > TARGET:
            chunks.append("\n\n".join(cur))
            carry, c = [], 0
            for x in reversed(cur):
                kx = ntok(x)
                if c + kx > OVERLAP: break
                carry.insert(0, x); c += kx
            cur, n = carry, c
        cur.append(b); n += k
    if cur: chunks.append("\n\n".join(cur))
    return chunks

rows = []
for path in sorted(CORPUS.glob("*.md")):
    source, text = read_doc(path)
    secs = [(p, f"{t}\n\n{b}" if t else b) for p, t, b in split_sections(text)]
    # 相鄰小節合併，直到接近 TARGET
    groups, cur_path, cur_txt, cur_n = [], None, [], 0
    for p, t in secs:
        k = ntok(t)
        if cur_txt and cur_n + k > TARGET:
            groups.append((cur_path, "\n\n".join(cur_txt))); cur_txt, cur_n = [], 0
        if not cur_txt: cur_path = p
        cur_txt.append(t); cur_n += k
    if cur_txt: groups.append((cur_path, "\n\n".join(cur_txt)))
    i = 0
    for p, t in groups:
        for c in pack(t):
            rows.append({"id": f"{path.stem}#{i:03d}", "doc": path.stem, "source": source,
                         "heading": p, "text": c, "n_tokens": ntok(c)})
            i += 1

with open(OUT, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

ns = [r["n_tokens"] for r in rows]
print(f"chunks: {len(rows)}  tokens mean/median/max: {statistics.mean(ns):.0f}/{statistics.median(ns):.0f}/{max(ns)}")
print(f"<50 tokens: {sum(n < 50 for n in ns)}   >450 tokens: {sum(n > 450 for n in ns)}")
print(dict(collections.Counter(r["doc"].split("__")[0] for r in rows)))
