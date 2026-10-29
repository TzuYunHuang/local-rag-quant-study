import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
d = json.load(open(ROOT/"review/grounding10_dump.json", encoding="utf-8"))
ZH = {
1: ("TFLite 的 dynamic range、full integer、float16 量化，在模型縮小幅度與硬體上如何比較？",
    "TFLite dynamic range 量化讓模型縮小 4 倍、加速 2x-3x，適用 CPU。Full integer 量化縮小 4 倍、加速 3x 以上，支援 CPU、Edge TPU 與 microcontroller。Float16 量化讓模型縮小 2 倍並可用 GPU 加速，在 GPU 上提供 2x-3x 加速。"),
2: ("ONNX Runtime 用什麼公式把量化值映射回浮點數？",
    "ONNX Runtime 使用 context 中提供的公式把量化值映射回浮點數。公式為：val_fp32 = scale * (val_quantized - zero_point)。其中 scale 與 zero_point 依資料範圍與量化範圍計算，如 context 所述。反向運算並未直接提供，但可用此反向公式由量化值推得原始浮點值。"),
3: ("為什麼 TFLite 8-bit 量化規格強制 weights 為對稱且 zero-point 等於 0？",
    "TFLite 8-bit 量化規格強制 weights 對稱且 zero-point 為 0，是為了避免 weight 的 zero-point 與 activation 值相乘所造成的無法避免的 runtime 成本。這讓量化盡可能有效率、避免不必要的運算負擔。對稱的 weights 讓 zero-point 值得以最佳化，這是取得額外精度的一種低成本方式。"),
4: ("為什麼 PyTorch 2 Export (PT2E) 量化流程宣稱比 FX Graph Mode Quantization 更好？",
    "PT2E 量化流程被宣稱比 FX Graph Mode Quantization 更好，因為它提供明顯更高的模型覆蓋率、更好的可程式化能力，以及更簡化的使用體驗。這來自於「預期此流程比 FX Graph Mode Quantization 有明顯更高的模型覆蓋率、更好的可程式化能力與更簡化的使用體驗」這段敘述。教學也提到 PT2E 流程可 export 的 construct 明顯更多，這是達到更高覆蓋率的關鍵因素。"),
5: ("torchao 的 QAT 有哪兩個步驟？各自做什麼？",
    "在 torchao 的 Quantization-Aware Training (QAT) 中，兩個步驟是：\n\n1. **Prepare**：在 linear layer 插入 fake quantization 運算。這會修改模型，加入在訓練階段模擬 quantization 與 dequantization 行為的運算。\n\n2. **Convert**：訓練後把模擬的 quantization 運算轉成真正的 quantization 與 dequantization 運算。最後一步會把 dequantization 運算與 linear layer 融合，模型即可進行量化。\n\n這兩步都是讓模型在訓練中與訓練後能處理量化的關鍵，確保最終量化模型能達到更高準確度與更低 perplexity。"),
6: ("為什麼 TFLite 8-bit 量化規格強制 weights 為對稱且 zero-point 等於 0？",
    "TFLite 8-bit 量化規格強制 weights 對稱且 zero-point 為 0，是為了避免 weight 的 zero-point 與 activation 值相乘的成本。這項最佳化對量化公式的數學是必要的，使 \\(\\sum_{i=0}^{n} q_{a}^{(i)} z_{b}\\) 這一項只需計算一次，因而去除不必要的 runtime 成本。這項最佳化對維持量化模型的效能很重要，尤其是在計算這一項成本很高的情境。"),
7: ("ONNX Runtime 中 dynamic 與 static quantization 有何不同？",
    "Dynamic quantization 在量化過程中計算 activation 的量化參數（scale 與 zero point），這會增加 inference 成本，但通常比 static quantization 準確度更高。Static quantization 則是用 calibration data 跑模型來計算這些參數，再把它們寫成量化模型中的常數，供所有輸入使用。"),
8: ("在 TFLite 量化規格中，per-tensor 與 per-axis 量化有何不同？",
    "Per-tensor 量化對整個 tensor 套用單一的 scale 與 zero-point，而 per-axis 量化是依每個 axis 套用這些值，具體是針對 `quantized_dimension` 中的 slice。Per-tensor 在整個 tensor 使用相同的 scale 與 zero-point，per-axis 則可對每個 axis（特別是指定的維度）套用不同的值。這讓 TFLite 的量化更細緻，能提升特定維度或 layer 的準確度。"),
9: ("TFLite 的 post-training quantization 方法中，哪一個建議作為起點？為什麼？",
    "建議作為 post-training quantization 起點的方法是 **Dynamic range quantization**。因為它能讓模型縮小 4 倍並加速 2x-3x，效益顯著。此外，它支援 on-the-fly 的 quantization 與 dequantization，可混合 quantized 與 floating-point kernel，對圖中不同部分提升效能。不過要注意確認量化模型的準確度，因為此方法不會重新訓練模型來補償量化。因此它適合作為評估量化效益的起點。"),
10: ("TFLite 的 integer quantization with float fallback 與 integer-only quantization 有何差異？",
    "差異在於產出的模型與不同硬體的相容性。Integer with float fallback 對沒有 integer 實作的運算使用 float 運算子，確保需要時能順利轉成 float。這讓模型相容於更廣泛的裝置與加速器，包括支援 float 輸入與輸出的裝置。Integer-only 量化則強制所有運算量化成 integer 型別，input 與 output 都使用 tf.int8 或 tf.uint8，因此模型只相容於支援 integer 輸入與輸出的硬體，例如 8-bit microcontroller 與 Coral Edge TPU。"),
}
def quote(zh): return "\n".join("> "+l if l.strip() else ">" for l in zh.split("\n"))
def fix(a): return a + ("\n```" if a.count("```") % 2 else "")
o = ["# 依據抽查（10 題）\n",
"**怎麼判斷**：只看「答案裡的事實」能不能在該題的 Context 找到。\n\n- 有依據（1）：答案的每個事實都出自 Context（即使跟標準答案不同也算）\n- 沒依據（0）：答案含有 Context 沒有的內容（編造、外部知識、推論過頭）\n\n**不用管答案對不對**，只看有沒有憑空多出來的東西。Context 為英文原文，未翻譯。\n\n---\n"]
for x in d:
    zq, za = ZH[x["n"]]
    o.append(f"## {x['n']}\n\n**Question**\n\n{x['question']}\n\n{quote(zq)}\n\n**Answer**\n\n{fix(x['answer'])}\n\n{quote(za)}\n\n**Context**（{len(x['context'])} 段）\n")
    for cid, h, t in x["context"]:
        o.append(f"`{cid}`\n\n````\n{t}\n````\n")
    o.append("---\n")
(ROOT/"review/grounding10.md").write_text("\n".join(o), encoding="utf-8")
print(sum(len(s) for s in o))
