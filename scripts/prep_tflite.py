import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "corpus_raw/tflite_src/tensorflow/lite/g3doc/performance"
OUT = ROOT / "corpus"
BASE = "https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/"
FILES = [
    "post_training_quantization.md", "quantization_spec.md", "model_optimization.md",
    "post_training_quant.ipynb", "post_training_integer_quant.ipynb",
    "post_training_float16_quant.ipynb", "post_training_integer_quant_16x8.ipynb",
    "quantization_debugger.ipynb",
]

def nb_to_md(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for c in nb["cells"]:
        text = "".join(c["source"]).strip()
        if not text or "Licensed under the Apache License" in text:
            continue
        parts.append(text if c["cell_type"] == "markdown" else f"```python\n{text}\n```")
    return "\n\n".join(parts)

OUT.mkdir(exist_ok=True)
for name in FILES:
    p = SRC / name
    body = nb_to_md(p) if name.endswith(".ipynb") else p.read_text(encoding="utf-8")
    (OUT / f"tflite__{p.stem}.md").write_text(
        f"<!-- source: {BASE}{name} -->\n\n{body}\n", encoding="utf-8")
    print(f"ok  tflite__{p.stem}.md  {len(body)} chars")
