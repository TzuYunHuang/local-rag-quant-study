import pathlib, re, pypandoc

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "corpus_raw"
OUT = ROOT / "corpus"

TF = ("tflite", RAW / "tflite_src/tensorflow/lite/g3doc",
      "https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/")
AO = ("pytorch", RAW / "torchao_src/docs/source",
      "https://github.com/pytorch/ao/blob/main/docs/source/")
OR = ("ort", RAW / "ort_src/docs",
      "https://github.com/microsoft/onnxruntime/blob/gh-pages/docs/")

FILES = [
    (TF, "models/convert/convert_models.md"),
    (TF, "models/convert/rnn.md"),
    (TF, "android/delegates/hexagon.md"),
    (AO, "pt2e_quantization/pt2e_quant_x86_inductor.rst"),
    (AO, "eager_tutorials/subclass_basic.rst"),
    (AO, "eager_tutorials/serving.rst"),
    (AO, "eager_tutorials/torchao_vllm_integration.md"),
    (OR, "execution-providers/QNN-ExecutionProvider.md"),
    (OR, "execution-providers/NNAPI-ExecutionProvider.md"),
    (OR, "execution-providers/Xnnpack-ExecutionProvider.md"),
]

def clean_jekyll(text):
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
    text = re.sub(r"^\s*\*\s*TOC placeholder\s*$", "", text, flags=re.M)
    text = re.sub(r"^## Contents\s*$", "", text, flags=re.M)
    h1 = re.match(r"\s*# (.*)\n", text)
    if title and h1 and h1.group(1).strip().lower() == title.lower():
        text = text[h1.end():]
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return (f"# {title}\n\n" if title else "") + text

OUT.mkdir(exist_ok=True)
for (prefix, root, base), rel in FILES:
    p = root / rel
    if p.suffix == ".rst":
        body = pypandoc.convert_file(str(p), "gfm", format="rst")
    else:
        body = p.read_text(encoding="utf-8")
    if prefix == "ort":
        body = clean_jekyll(body)
    (OUT / f"{prefix}__{p.stem}.md").write_text(
        f"<!-- source: {base}{rel} -->\n\n{body}\n", encoding="utf-8")
    print(f"ok  {prefix}__{p.stem}.md  {len(body)} chars")
