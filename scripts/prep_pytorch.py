import pathlib, shutil, pypandoc

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "corpus_raw"
OUT = ROOT / "corpus"

# (本機路徑, 來源網址)
PT = "https://github.com/pytorch/pytorch/blob/main/docs/source/"
AO = "https://github.com/pytorch/ao/blob/main/docs/source/"
FILES = [
    (RAW / "pytorch_src/docs/source/quantization.md", PT + "quantization.md"),
] + [
    (RAW / "torchao_src/docs/source" / rel, AO + rel) for rel in [
        "pt2e_quantization/pt2e_quant_ptq.rst",
        "pt2e_quantization/pt2e_quant_qat.rst",
        "pt2e_quantization/pt2e_quantizer.rst",
        "contributing/quantization_overview.rst",
        "eager_tutorials/static_quantization.rst",
        "eager_tutorials/first_quantization_example.rst",
        "eager_tutorials/finetuning.rst",
        "workflows/qat.md",
        "workflows/inference.md",
    ]
]

OUT.mkdir(exist_ok=True)
for path, url in FILES:
    if path.suffix == ".rst":
        body = pypandoc.convert_file(str(path), "gfm", format="rst")
    else:
        body = path.read_text(encoding="utf-8")
    (OUT / f"pytorch__{path.stem}.md").write_text(
        f"<!-- source: {url} -->\n\n{body}\n", encoding="utf-8")
    print(f"ok  pytorch__{path.stem}.md  {len(body)} chars")
