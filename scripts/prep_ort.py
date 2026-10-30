import pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "corpus_raw/ort_src/docs/performance"
OUT = ROOT / "corpus"
BASE = "https://github.com/microsoft/onnxruntime/blob/gh-pages/docs/performance/"
FILES = [
    "model-optimizations/quantization.md",
    "model-optimizations/float16.md",
    "model-optimizations/graph-optimizations.md",
]

def clean(text):
    title = None
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if m:
        t = re.search(r"^title:\s*(.+)$", m.group(1), re.M)
        title = t.group(1).strip().strip('"\'') if t else None
        text = text[m.end():]
    text = re.sub(r"^\{:[^}]*\}\s*$", "", text, flags=re.M)
    text = re.sub(r"^.*\{:toc\}.*$", "", text, flags=re.M)
    text = re.sub(r"^\d+\.\s*TOC\s*$", "", text, flags=re.M)
    text = re.sub(r"\{%.*?%\}", "", text, flags=re.S)
    text = re.sub(r"^## Contents\s*\n+\s*\*\s*TOC placeholder\s*$", "", text, flags=re.M)
    text = re.sub(r"^\s*\*\s*TOC placeholder\s*$", "", text, flags=re.M)
    text = re.sub(r"^# .*\n", "", text, count=1, flags=re.M)   # duplicate H1 already present in the body
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return (f"# {title}\n\n" if title else "") + text

OUT.mkdir(exist_ok=True)
for rel in FILES:
    p = SRC / rel
    body = clean(p.read_text(encoding="utf-8"))
    (OUT / f"ort__{p.stem}.md").write_text(
        f"<!-- source: {BASE}{rel} -->\n\n{body}\n", encoding="utf-8")
    print(f"ok  ort__{p.stem}.md  {len(body)} chars")
