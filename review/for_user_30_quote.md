# D9 覆核（中英對照）：請對每個答案打 0／1／2 分

- 2：與標準答案的關鍵事實都一致，沒有錯誤內容
- 1：對了一部分（缺關鍵點，或夾帶小錯）
- 0：錯誤、離題、答案被截斷，或有答案題卻拒答
- 無答案題（標準答案為 Not stated）：正確拒答＝2；拒答但編造縮寫全名＝1；直接作答＝0

回覆格式範例：`1=2, 2=0, 3=1, ...`（30 題）

說明：中文為方便對照的翻譯，模型答案的錯誤也照實翻出，沒有修正。請以英文原文為準評分。

**1. [fact] Where is PyTorch centralizing its quantization development?**

> PyTorch 把量化開發集中在哪裡？

**標準答案：**
In the torchao project (pytorch/ao); the old torch.ao.quantization flows are being migrated there.

> 在 torchao 專案（pytorch/ao）；舊的 torch.ao.quantization 流程正在遷移到那裡。

**模型答案：**
PyTorch is centralizing its quantization development through the introduction of the PyTorch 2 Export Quantization flow, which includes support for both static and dynamic quantization, and utilizes the torch.export mechanism to capture the model into a graph. This flow also integrates with the TorchInductor compiler backend, allowing for the generation of a quantized model customized for the x86 inductor backend. The flow supports programmability and simplifies the user experience by providing a single instance for both backend and user interaction.

> PyTorch 透過引入 PyTorch 2 Export Quantization 流程來集中其量化開發，該流程支援靜態與動態量化，並利用 torch.export 機制把模型擷取成圖。這個流程也與 TorchInductor 編譯器後端整合，可產生為 x86 inductor 後端客製的量化模型。此流程支援可程式化性，並透過提供單一實例供後端與使用者互動來簡化使用者體驗。

---

**2. [compare] What data does each TFLite quantization technique need: float16, dynamic range, integer, and quantization-aware training?**

> TFLite 各量化技術需要什麼資料：float16、dynamic range、integer 與 quantization-aware training？

**標準答案：**
Float16 and dynamic range need no data; integer quantization needs an unlabelled representative sample; quantization-aware training needs labelled training data.

> Float16 與 dynamic range 不需要資料；integer 量化需要未標註的代表性樣本；quantization-aware training 需要已標註的訓練資料。

**模型答案：**
For float16 quantization, the model weights are quantized to float16. For dynamic range quantization, activations are dynamically quantized to 8 bits during inference. For integer only quantization, the model weights are quantized to 8-bit integers and activations to 16-bit integers. Quantization-aware training requires retraining the model to compensate for quantization-induced errors.

> float16 量化是把模型權重量化成 float16。dynamic range 量化是在推論時把 activations 動態量化成 8 bits。integer-only 量化是把模型權重量化成 8-bit 整數、activations 量化成 16-bit 整數。Quantization-aware training 需要重新訓練模型，以補償量化引起的誤差。

---

**3. [compare] How should users migrate from PyTorch's eager mode and FX graph mode quantization flows?**

> 使用者應如何從 PyTorch 的 eager mode 與 FX graph mode 量化流程遷移？

**標準答案：**
Eager mode quantization users should migrate to torchao's quantize_ API; FX graph mode users should migrate to torchao's pt2e quantization API (prepare_pt2e and convert_pt2e). pt2e quantization itself has moved to torchao.

> Eager mode 量化的使用者應遷移到 torchao 的 quantize_ API；FX graph mode 的使用者應遷移到 torchao 的 pt2e 量化 API（prepare_pt2e 與 convert_pt2e）。pt2e 量化本身已搬到 torchao。

**模型答案：**
To migrate from PyTorch's eager mode and FX graph mode quantization flows, users should follow the tutorial on PyTorch 2 Export Quantization. This tutorial explains the steps to generate a quantized model using the PyTorch 2 Export Quantization flow, which is designed for the x86 Inductor backend. The tutorial provides a detailed workflow that includes capturing the model, applying quantization, and lowering the model into the Inductor. Users are advised to implement a `Quantizer` for their specific backend, as the model produced at this stage is not the final model that runs on the device. The tutorial also provides guidance on debugging and performance evaluation of the quantized models. Therefore, users should migrate to the PyTorch 2 Export Quantization flow as a more advanced and flexible approach for quantization.

> 要從 PyTorch 的 eager mode 與 FX graph mode 量化流程遷移，使用者應該照著 PyTorch 2 Export Quantization 的教學。這份教學說明如何使用 PyTorch 2 Export Quantization 流程產生量化模型，該流程是為 x86 Inductor 後端設計的。教學提供完整的工作流程，包括擷取模型、套用量化，以及把模型降階（lowering）到 Inductor。建議使用者為自己的後端實作一個 `Quantizer`，因為這個階段產生的模型還不是在裝置上執行的最終模型。教學也提供量化模型的除錯與效能評估指引。因此，使用者應遷移到 PyTorch 2 Export Quantization 流程，作為更先進、更靈活的量化方式。

---

**4. [unanswerable] How do I run a GGUF Q4_K_M model with llama.cpp?**

> 我要如何用 llama.cpp 執行 GGUF Q4_K_M 模型？

**標準答案：**
Not stated in the documents.

> 文件中沒有說明。

**模型答案：**
To run a GGUF Q4_K_M model with `llama.cpp` on a mobile device, you would need to follow the instructions provided for iOS devices. According to the context, this can be done using the `.pte` file generated from the model. The instructions for iOS devices are detailed in the provided link. Therefore, you would need to follow the iOS device instructions to run the model.

> 要在行動裝置上用 `llama.cpp` 執行 GGUF Q4_K_M 模型，你需要依照為 iOS 裝置提供的說明。根據上下文，這可以透過從模型產生的 `.pte` 檔案來完成。iOS 裝置的說明在所提供的連結中有詳細介紹。因此，你需要依照 iOS 裝置的說明來執行模型。

---

**5. [fact] Which TFLite post-training quantization method is recommended as a starting point, and why?**

> TFLite 的訓練後量化方法中，建議以哪一種作為起點？為什麼？

**標準答案：**
Dynamic range quantization, because it reduces memory usage and speeds up computation without requiring a representative dataset for calibration.

> Dynamic range 量化，因為它能減少記憶體用量並加快運算，而且不需要代表性資料集來校準。

**模型答案：**
Dynamic range quantization is recommended as a starting point because it reduces memory usage and speeds up computation, providing 8-bit precision with only the weights quantized at conversion time, and it is a recommended approach for reduced memory usage and faster computation. However, it still requires floating-point storage for the outputs.

> 建議以 dynamic range 量化作為起點，因為它能減少記憶體用量並加快運算，在轉換時只量化權重、提供 8-bit 精度，而且它是減少記憶體用量與加快運算的建議做法。不過，輸出仍需要以浮點數儲存。

---

**6. [compare] How do dynamic quantization and weight-only quantization differ in torchao's quantized inference?**

> 在 torchao 的量化推論中，dynamic quantization 與 weight-only quantization 有何不同？

**標準答案：**
Dynamic quantization converts both the input activations and the weights to low precision (for example fp8 rowwise) before the matmul. Weight-only quantization (for example int4) quantizes only the weight and keeps the activations in high precision.

> Dynamic quantization 在矩陣乘法前，把輸入 activations 與權重都轉成低精度（例如 fp8 rowwise）。Weight-only quantization（例如 int4）只量化權重，activations 保持高精度。

**模型答案：**
In torchao's quantized inference, dynamic quantization involves converting the input and weight tensors to a quantized format (like fp8) before performing the matrix multiplication. Weight-only quantization, on the other hand, only converts the weight tensor to a quantized format (like int4) while keeping the input tensor in its original precision.

> 在 torchao 的量化推論中，dynamic quantization 是在執行矩陣乘法前，把輸入與權重 tensor 轉成量化格式（如 fp8）。Weight-only quantization 則只把權重 tensor 轉成量化格式（如 int4），輸入 tensor 保持原本精度。

---

**7. [compare] What data does each TFLite quantization technique need: float16, dynamic range, integer, and quantization-aware training?**

> TFLite 各量化技術需要什麼資料：float16、dynamic range、integer 與 quantization-aware training？

**標準答案：**
Float16 and dynamic range need no data; integer quantization needs an unlabelled representative sample; quantization-aware training needs labelled training data.

> Float16 與 dynamic range 不需要資料；integer 量化需要未標註的代表性樣本；quantization-aware training 需要已標註的訓練資料。

**模型答案：**
For Post-training float16 quantization, no data is needed. For Post-training dynamic range quantization, no data is also needed. For Post-training integer quantization, an unlabelled representative sample is required. For quantization-aware training, labelled training data is needed.

> 訓練後 float16 量化不需要資料。訓練後 dynamic range 量化同樣不需要資料。訓練後 integer 量化需要未標註的代表性樣本。Quantization-aware training 需要已標註的訓練資料。

---

**8. [number] In the TFLite tutorial, how much does dynamic range quantization shrink the ResNet v2 101 model?**

> 在 TFLite 教學中，dynamic range 量化把 ResNet v2 101 模型縮小了多少？

**標準答案：**
From 171 MB to 43 MB, with the same top-1 accuracy of 76.8.

> 從 171 MB 縮小到 43 MB，top-1 準確率同為 76.8。

**模型答案：**
I cannot find this in the provided documents.

> 我在提供的文件中找不到這個資訊。

---

**9. [fact] What formula does ONNX Runtime use to map a quantized value back to a floating point value?**

> ONNX Runtime 用什麼公式把量化值對應回浮點數值？

**標準答案：**
val_fp32 = scale * (val_quantized - zero_point).

> val_fp32 = scale * (val_quantized - zero_point)。

**模型答案：**
ONNX Runtime uses the formula provided in the context to map a quantized value back to a floating point value. The formula is:

```
val_fp32 = scale * (val_quantized - zero_point)
```

Where `scale` and `zero_point` are calculated based on the data range and quantization range, as detailed in the context. The inverse operation is not directly provided, but the original floating point value can be derived from the quantized value using this inverse formula.

> ONNX Runtime 使用上下文中提供的公式把量化值對應回浮點數值。公式是：
>
> ```
> val_fp32 = scale * (val_quantized - zero_point)
> ```
>
> 其中 `scale` 與 `zero_point` 依資料範圍與量化範圍計算，如上下文所述。反向運算沒有直接提供，但可以用這個反向公式從量化值推得原本的浮點數值。

---

**10. [fact] What known limitation does ONNX Runtime have when optimizing large models during quantization pre-processing?**

> ONNX Runtime 在量化前處理中最佳化大型模型時，有什麼已知限制？

**標準答案：**
Model optimization cannot output a model larger than 2GB, so it must be skipped for large models.

> 模型最佳化無法輸出大於 2GB 的模型，因此大型模型必須跳過這個步驟。

**模型答案：**
ONNX Runtime has a known limitation where model optimization during quantization pre-processing can only output a model size greater than 2GB. For large models, optimization must be skipped.

> ONNX Runtime 有一個已知限制：量化前處理中的模型最佳化只能輸出大於 2GB 的模型。對大型模型來說，必須跳過最佳化。

---

**11. [fact] What is the prerequisite for using the PT2E post-training quantization flow?**

> 使用 PT2E 訓練後量化流程的先決條件是什麼？

**標準答案：**
The model must be exportable with torch.export.export.

> 模型必須能用 torch.export.export 匯出。

**模型答案：**
Exportable by `torch.export.export` is a prerequisite for using the PT2E post-training quantization flow.

> 能夠以 `torch.export.export` 匯出，是使用 PT2E 訓練後量化流程的先決條件。

---

**12. [compare] How do dynamic quantization and weight-only quantization differ in torchao's quantized inference?**

> 在 torchao 的量化推論中，dynamic quantization 與 weight-only quantization 有何不同？

**標準答案：**
Dynamic quantization converts both the input activations and the weights to low precision (for example fp8 rowwise) before the matmul. Weight-only quantization (for example int4) quantizes only the weight and keeps the activations in high precision.

> Dynamic quantization 在矩陣乘法前，把輸入 activations 與權重都轉成低精度（例如 fp8 rowwise）。Weight-only quantization（例如 int4）只量化權重，activations 保持高精度。

**模型答案：**
Dynamic quantization and weight-only quantization in torchao's quantized inference differ in how they handle the weights. In dynamic quantization, the weights are converted to a lower precision format (like fp8 in the example), while in weight-only quantization, only the weights are quantized to a lower precision (like int4 in the example), leaving the inputs at their full precision.

> 在 torchao 的量化推論中，dynamic quantization 與 weight-only quantization 的差別在於處理權重的方式。在 dynamic quantization 中，權重被轉成較低精度格式（如範例中的 fp8）；在 weight-only quantization 中，只有權重被量化成較低精度（如範例中的 int4），輸入保持完整精度。

---

**13. [fact] In the PT2E QAT flow, why should you not call model.eval() or model.train() after program capture?**

> 在 PT2E QAT 流程中，為什麼擷取程式（program capture）之後不應呼叫 model.eval() 或 model.train()？

**標準答案：**
These methods no longer correctly change the behavior of ops like dropout and batch normalization after export; use move_exported_model_to_eval (and move_exported_model_to_train) instead.

> 匯出之後，這些方法不再能正確改變 dropout 與 batch normalization 等運算的行為；應改用 move_exported_model_to_eval（以及 move_exported_model_to_train）。

**模型答案：**
In the PyTorch 2 Export Quantization-Aware Training (QAT) flow, you should not call `model.eval()` or `model.train()` after program capture because these methods no longer correctly change the behavior of certain operations like dropout and batch normalization. Instead, you must use the provided functions to ensure these operations behave correctly in the evaluation or training modes, respectively. This is to maintain the integrity of the operations' behavior during inference.

> 在 PyTorch 2 Export Quantization-Aware Training（QAT）流程中，擷取程式之後不應呼叫 `model.eval()` 或 `model.train()`，因為這些方法不再能正確改變 dropout 與 batch normalization 等某些運算的行為。你必須改用所提供的函式，確保這些運算分別在評估或訓練模式下行為正確。這是為了在推論時維持這些運算行為的完整性。

---

**14. [fact] In ONNX Runtime quantization, why must the floating point zero be exactly representable in the quantization space?**

> 在 ONNX Runtime 量化中，為什麼浮點數的 0 必須能在量化空間中被精確表示？

**標準答案：**
Because zero padding is used in many CNNs. If 0 cannot be represented uniquely after quantization, it causes accuracy errors.

> 因為許多 CNN 會使用 zero padding。如果量化後 0 無法被唯一表示，就會造成準確度誤差。

**模型答案：**
In ONNX Runtime quantization, the floating point zero must be exactly representable in the quantization space to avoid accuracy errors. If the floating point zero value cannot be uniquely represented after quantization, it can lead to inaccuracies in the model's performance.

> 在 ONNX Runtime 量化中，浮點數的 0 必須能在量化空間中被精確表示，以避免準確度誤差。如果量化後浮點數 0 的值無法被唯一表示，可能導致模型表現不準確。

---

**15. [compare] What is the difference between TFLite integer quantization with float fallback and integer-only quantization?**

> TFLite 的「帶 float fallback 的整數量化」與「integer-only 量化」有何不同？

**標準答案：**
With float fallback, ops without an integer implementation stay in float and the input and output stay float, so the model is not compatible with integer-only devices such as 8-bit microcontrollers or Coral Edge TPU. Integer-only enforces integer quantization for all ops including input and output, for those devices.

> 使用 float fallback 時，沒有整數實作的運算會保持 float，輸入與輸出也保持 float，因此模型不相容於 integer-only 裝置（如 8-bit 微控制器或 Coral Edge TPU）。Integer-only 則對所有運算（包含輸入與輸出）強制整數量化，以適用於那些裝置。

**模型答案：**
The difference between TFLite integer quantization with float fallback and integer-only quantization lies in the compatibility with devices and accelerators. Integer with float fallback allows the use of float operators when integer implementations are not available, making it smoother for conversion but not compatible with devices that require integer inputs and outputs. In contrast, integer-only quantization ensures that the model is fully quantized to integers, including the input and output, making it compatible with devices like 8-bit microcontrollers and Coral Edge TPUs.

> TFLite 的 float fallback 整數量化與 integer-only 量化的差別，在於與裝置及加速器的相容性。帶 float fallback 的整數量化在沒有整數實作時允許使用 float 運算子，讓轉換更順利，但不相容於需要整數輸入與輸出的裝置。相對地，integer-only 量化確保模型完全量化成整數（包含輸入與輸出），因此相容於 8-bit 微控制器與 Coral Edge TPU 這類裝置。

---

**16. [number] What CER does DeepSpeech 0.5.1 reach with float32, int8 and int16 activations in the TFLite int16 table?**

> 在 TFLite int16 表格中，DeepSpeech 0.5.1 用 float32、int8 與 int16 activations 的 CER 各是多少？

**標準答案：**
6.13% with float32 activations, 43.67% with int8 activations, and 6.52% with int16 activations.

> float32 activations 為 6.13%，int8 activations 為 43.67%，int16 activations 為 6.52%。

**模型答案：**
The CER (Character Error Rate) for DeepSpeech 0.5.1 with int8 activations is 43.67%. For int16 activations, it is 6.52%. The table does not provide a float32 activation CER for DeepSpeech 0.5.1.

> DeepSpeech 0.5.1 在 int8 activations 下的 CER（字元錯誤率）是 43.67%。在 int16 activations 下是 6.52%。表格沒有提供 DeepSpeech 0.5.1 在 float32 activation 下的 CER。

---

**17. [fact] What are the two steps of QAT in torchao, and what does each do?**

> torchao 中 QAT 的兩個步驟是什麼？各自做什麼？

**標準答案：**
Prepare inserts fake quantize operations into linear layers; convert turns the fake quantize operations into actual quantize and dequantize operations after training.

> Prepare 在 linear 層插入 fake quantize 運算；convert 在訓練後把 fake quantize 運算轉成實際的 quantize 與 dequantize 運算。

**模型答案：**
In torchao, the two steps of Quantization-Aware Training (QAT) are:

1. **Prepare**: This step inserts fake quantize operations into linear layers of the model. This allows the model to be trained as if it were quantized, but without actually changing the data types of the weights and activations.

2. **Convert**: After training, this step transforms the fake quantize operations into actual quantize and dequantize operations. This step is crucial for producing a quantized model, as it replaces the fake quantize operations with the actual quantization and dequantization that would be used in a post-training quantization process.

> 在 torchao 中，Quantization-Aware Training（QAT）的兩個步驟是：
>
> 1. **Prepare**：這個步驟在模型的 linear 層插入 fake quantize 運算。這讓模型可以像被量化一樣進行訓練，但不會真的改變權重與 activations 的資料型別。
>
> 2. **Convert**：訓練後，這個步驟把 fake quantize 運算轉成實際的 quantize 與 dequantize 運算。這個步驟對產生量化模型很關鍵，因為它把 fake quantize 運算換成訓練後量化流程中會用到的實際量化與反量化。

---

**18. [compare] How does float16 conversion in TFLite compare with float16 conversion in ONNX Runtime?**

> TFLite 的 float16 轉換與 ONNX Runtime 的 float16 轉換如何比較？

**標準答案：**
Both can shrink the model by up to half with small accuracy loss and benefit GPUs. TFLite float16 quantizes weights and dequantizes them to float32 on CPU by default, while the GPU delegate runs float16 directly. ONNX Runtime converts the model with a float16 tool, needs no tuning data, may need op block lists or mixed precision, and the CPU version does not support float16 ops.

> 兩者都最多可把模型縮小一半，準確度損失小，且對 GPU 有益。TFLite 的 float16 會量化權重，並在 CPU 上預設把它們反量化成 float32，而 GPU delegate 則直接執行 float16。ONNX Runtime 用 float16 工具轉換模型，不需要調校資料，可能需要運算黑名單（op block list）或混合精度，且 CPU 版本不支援 float16 運算。

**模型答案：**
The provided context does not contain information that directly compares the float16 conversion process in TFLite to that in ONNX Runtime. The information given is specific to the float16 quantization process in TensorFlow and ONNX, and does not provide a direct comparison between the two. Therefore, I cannot find this information in the provided documents.

> 所提供的上下文沒有直接比較 TFLite 與 ONNX Runtime 的 float16 轉換流程的資訊。給定的資訊分別針對 TensorFlow 與 ONNX 的 float16 量化流程，並沒有提供兩者的直接比較。因此，我在提供的文件中找不到這個資訊。

---

**19. [unanswerable] What is the latency of an int8 MobileNetV2 in TFLite on a Raspberry Pi 4?**

> int8 MobileNetV2 在 Raspberry Pi 4 上用 TFLite 的延遲是多少？

**標準答案：**
Not stated in the documents.

> 文件中沒有說明。

**模型答案：**
The provided context does not contain information about the latency of an int8 MobileNetV2 in TFLite on a Raspberry Pi 4. The details about the latency of MobileNetV2 on a Raspberry Pi 4, particularly in the context of int8 (intelligent) quantization, are not included in the given information. Therefore, I cannot find this information in the provided documents.

> 所提供的上下文沒有關於 int8 MobileNetV2 在 Raspberry Pi 4 上用 TFLite 的延遲資訊。關於 MobileNetV2 在 Raspberry Pi 4 上的延遲細節，特別是在 int8（intelligent，智慧）量化的情境下，並未包含在給定資訊中。因此，我在提供的文件中找不到這個資訊。

---

**20. [fact] Why does TFLite full integer quantization require a representative dataset?**

> 為什麼 TFLite 的 full integer 量化需要代表性資料集？

**標準答案：**
Variable tensors such as inputs, activations and outputs cannot be calibrated unless inference is run, so the converter needs sample data to estimate their (min, max) ranges.

> 輸入、activations 與輸出這類可變 tensor，必須實際執行推論才能校準，所以轉換器需要樣本資料來估計它們的 (min, max) 範圍。

**模型答案：**
TFLite full integer quantization requires a representative dataset because variable tensors such as model input, activations, and model output cannot be calibrated with just the model's weights and biases. These tensors need to be calibrated through actual inference cycles, which necessitates a dataset to simulate this process. The representative dataset provides the necessary data points to estimate the range of these tensors, allowing for proper quantization.

> TFLite full integer 量化需要代表性資料集，因為模型輸入、activations 與模型輸出這類可變 tensor，無法只靠模型的權重與偏差來校準。這些 tensor 需要透過實際的推論週期來校準，因此需要一個資料集來模擬這個過程。代表性資料集提供估計這些 tensor 範圍所需的資料點，使量化得以正確進行。

---

**21. [number] What CER does DeepSpeech 0.5.1 reach with float32, int8 and int16 activations in the TFLite int16 table?**

> 在 TFLite int16 表格中，DeepSpeech 0.5.1 用 float32、int8 與 int16 activations 的 CER 各是多少？

**標準答案：**
6.13% with float32 activations, 43.67% with int8 activations, and 6.52% with int16 activations.

> float32 activations 為 6.13%，int8 activations 為 43.67%，int16 activations 為 6.52%。

**模型答案：**
I cannot find this in the provided documents. The CER (Categorical Cross-Entropy) values for DeepSpeech 0.5.1 with float32, int8, and int16 activations in the TFLite int16 table are not specified in the provided context.

> 我在提供的文件中找不到這個資訊。所提供的上下文沒有指明 TFLite int16 表格中 DeepSpeech 0.5.1 在 float32、int8 與 int16 activations 下的 CER（Categorical Cross-Entropy，類別交叉熵）數值。

---

**22. [fact] What known limitation does ONNX Runtime have when optimizing large models during quantization pre-processing?**

> ONNX Runtime 在量化前處理中最佳化大型模型時，有什麼已知限制？

**標準答案：**
Model optimization cannot output a model larger than 2GB, so it must be skipped for large models.

> 模型最佳化無法輸出大於 2GB 的模型，因此大型模型必須跳過這個步驟。

**模型答案：**
The known limitation of ONNX Runtime when optimizing large models during quantization pre-processing is that it cannot output a model size greater than 2GB. So for large models, optimization must be skipped. This is detailed in the provided context.

> ONNX Runtime 在量化前處理中最佳化大型模型時的已知限制，是它無法輸出大於 2GB 的模型。所以對大型模型來說，必須跳過最佳化。這在所提供的上下文中有詳細說明。

---

**23. [compare] How do the PT2E post-training quantization and PT2E QAT flows differ?**

> PT2E 訓練後量化與 PT2E QAT 流程有何不同？

**標準答案：**
Both capture the model with torch.export and end with convert_pt2e. PTQ uses prepare_pt2e followed by calibration, while QAT uses prepare_qat_pt2e followed by training, and after export QAT must use move_exported_model_to_eval instead of model.eval().

> 兩者都用 torch.export 擷取模型，並以 convert_pt2e 結束。PTQ 使用 prepare_pt2e 再做校準，QAT 則使用 prepare_qat_pt2e 再做訓練，而且匯出後 QAT 必須用 move_exported_model_to_eval，不能用 model.eval()。

**模型答案：**
The PT2E post-training quantization and PT2E QAT flows differ in the steps of quantization-aware training (QAT). In QAT, the quantization-aware training step involves using a custom quantizer to prepare the model for quantization, which allows for the model to be trained with quantization in mind. This step is not present in the standard PT2E flow, which directly quantizes the model after capturing the model's operations.

> PT2E 訓練後量化與 PT2E QAT 流程的差別，在於 quantization-aware training（QAT）的步驟。在 QAT 中，quantization-aware training 這一步使用自訂的 quantizer 為量化準備模型，使模型能在考慮量化的情況下訓練。標準的 PT2E 流程沒有這一步，它在擷取模型運算後直接量化模型。

---

**24. [compare] What is the difference between post-training quantization and quantization-aware training?**

> 訓練後量化與 quantization-aware training 有何不同？

**標準答案：**
PTQ quantizes an already-trained model without retraining and may lose accuracy, especially on small networks. QAT applies fake quantization during training or fine-tuning so the final quantized model has higher accuracy, but needs changes to the training process.

> PTQ 量化已訓練好的模型、不需重新訓練，但可能損失準確度，在小型網路上尤其明顯。QAT 在訓練或微調期間套用 fake quantization，讓最終量化模型有較高準確度，但需要修改訓練流程。

**模型答案：**
Post-training quantization and quantization-aware training (QAT) are both techniques used to optimize models for reduced size and improved performance, but they approach the process differently. Post-training quantization is a one-time process that converts an already-trained model into a quantized version, which can reduce the model size and improve performance, but it does not require any changes to the training process. In contrast, quantization-aware training involves modifying the training process to include quantization operations, which can lead to better accuracy and lower perplexities, but it requires a more complex and time-consuming training process. The decision between these two methods depends on the specific requirements of the application, such as the desired balance between model size, accuracy, and performance.

> 訓練後量化與 quantization-aware training（QAT）都是用來縮小模型、提升效能的技術，但處理方式不同。訓練後量化是一次性的過程，把已訓練好的模型轉成量化版本，可以縮小模型並提升效能，且不需要修改訓練流程。相對地，quantization-aware training 需要修改訓練流程以加入量化運算，可帶來更好的準確度與更低的 perplexity，但需要更複雜且更耗時的訓練過程。兩種方法的選擇取決於應用的具體需求，例如在模型大小、準確度與效能之間想要的平衡。

---

**25. [compare] What is the difference between the QOperator and QDQ formats for quantized ONNX models?**

> 量化 ONNX 模型的 QOperator 與 QDQ 格式有何不同？

**標準答案：**
QOperator uses dedicated quantized operators such as QLinearConv and MatMulInteger. QDQ inserts DeQuantizeLinear(QuantizeLinear(tensor)) between the original operators to simulate quantization and dequantization.

> QOperator 使用專用的量化運算子，如 QLinearConv 與 MatMulInteger。QDQ 在原本的運算子之間插入 DeQuantizeLinear(QuantizeLinear(tensor))，以模擬量化與反量化。

**模型答案：**
The QOperator format represents quantized ONNX models where each quantized operator has its own ONNX definition, such as QLinearConv, MatMulInteger, etc. In contrast, the QDQ format inserts DeQuantizeLinear(QuantizeLinear(tensor)) between the original operators to simulate the quantization and dequantization process. In static quantization, the QuantizeLinear and DeQuantizeLinear operators carry quantization parameters, while in dynamic quantization, a ComputeQuantizationParameters function proto is used to calculate these parameters on the fly.

> QOperator 格式表示的量化 ONNX 模型中，每個量化運算子都有自己的 ONNX 定義，例如 QLinearConv、MatMulInteger 等。相對地，QDQ 格式在原本的運算子之間插入 DeQuantizeLinear(QuantizeLinear(tensor))，以模擬量化與反量化過程。在 static 量化中，QuantizeLinear 與 DeQuantizeLinear 運算子帶有量化參數；在 dynamic 量化中，則用 ComputeQuantizationParameters 函式原型（function proto）即時計算這些參數。

---

**26. [unanswerable] How does the AWQ algorithm decide which weight channels to protect from quantization error?**

> AWQ 演算法如何決定要保護哪些權重通道免受量化誤差影響？

**標準答案：**
Not stated in the documents.

> 文件中沒有說明。

**模型答案：**
The provided context does not explicitly detail how the AWQ (Weight Quantization) algorithm decides which weight channels to protect from quantization error. The information about the AWQ algorithm and its decision-making process for protecting specific weight channels is not provided in the given context. Therefore, I cannot find this information in the provided documents.

> 所提供的上下文沒有明確說明 AWQ（Weight Quantization，權重量化）演算法如何決定要保護哪些權重通道免受量化誤差影響。關於 AWQ 演算法及其保護特定權重通道的決策過程，給定的上下文中沒有提供資訊。因此，我在提供的文件中找不到這個資訊。

---

**27. [fact] What is the prerequisite for using the PT2E post-training quantization flow?**

> 使用 PT2E 訓練後量化流程的先決條件是什麼？

**標準答案：**
The model must be exportable with torch.export.export.

> 模型必須能用 torch.export.export 匯出。

**模型答案：**
The prerequisite for using the PT2E post-training quantization flow is having TensorFlow 1.15 or higher. This is mentioned in the context provided in note [1].

> 使用 PT2E 訓練後量化流程的先決條件是 TensorFlow 1.15 或更高版本。這在所提供上下文的註記 [1] 中有提到。

---

**28. [compare] How do dynamic and static quantization differ in ONNX Runtime?**

> 在 ONNX Runtime 中，dynamic 與 static 量化有何不同？

**標準答案：**
Dynamic quantization computes activation scale and zero point at inference time, which adds cost but usually gives higher accuracy. Static quantization runs calibration data beforehand and stores the activation parameters as constants in the model.

> Dynamic 量化在推論時計算 activation 的 scale 與 zero point，會增加成本但通常準確度較高。Static 量化事先用校準資料跑過，並把 activation 參數當作常數存在模型裡。

**模型答案：**
Dynamic quantization calculates quantization parameters (scale and zero point) for activations during the quantization process, which increases inference cost but usually achieves higher accuracy compared to static quantization. Static quantization, on the other hand, involves running the model with calibration data to compute these parameters, which are then written as constants in the quantized model and used for all inputs.

> Dynamic 量化在量化過程中計算 activations 的量化參數（scale 與 zero point），這會增加推論成本，但通常比 static 量化有更高的準確度。Static 量化則是用校準資料執行模型來計算這些參數，再把它們當作常數寫入量化後的模型，並用於所有輸入。

---

**29. [number] What F1 does MobileBert get with float32 activations versus int8 activations in the TFLite int16 table?**

> 在 TFLite int16 表格中，MobileBert 用 float32 activations 與 int8 activations 的 F1 各是多少？

**標準答案：**
F1 88.81 with float32 activations and 2.08 with int8 activations (88.73 with int16 activations).

> float32 activations 的 F1 為 88.81，int8 activations 為 2.08（int16 activations 為 88.73）。

**模型答案：**
The table provided does not contain the specific F1 (Exact match) F1(Exact match) accuracy values for MobileBert with float32 activations versus int8 activations in the TFLite int16 table. The provided data only includes the F1 accuracy for MobileBert with int8 activations, not the comparison with float32 activations. Therefore, I cannot find this information in the provided documents.

> 所提供的表格沒有包含 TFLite int16 表格中 MobileBert 在 float32 activations 與 int8 activations 下的具體 F1（Exact match）F1(Exact match) 準確度數值（原文此處重複了一次）。所提供的資料只包含 MobileBert 在 int8 activations 下的 F1 準確度，沒有與 float32 activations 的比較。因此，我在提供的文件中找不到這個資訊。

---

**30. [compare] How do TFLite dynamic range, full integer and float16 quantization compare in size reduction and hardware?**

> TFLite 的 dynamic range、full integer 與 float16 量化，在縮小模型與硬體方面如何比較？

**標準答案：**
Dynamic range: 4x smaller, 2x-3x speedup, CPU. Full integer: 4x smaller, 3x+ speedup, CPU, Edge TPU and microcontrollers. Float16: 2x smaller with GPU acceleration, CPU and GPU.

> Dynamic range：縮小 4 倍、加速 2 到 3 倍、CPU。Full integer：縮小 4 倍、加速 3 倍以上、CPU、Edge TPU 與微控制器。Float16：縮小 2 倍、可用 GPU 加速、CPU 與 GPU。

**模型答案：**
TFLite dynamic range quantization results in the smallest model size, though at the expense of a moderate increase in latency. It is best for CPU and hardware accelerators that can handle the increased latency. The float16 quantization provides the smallest model size, though at the cost of a 2x increase in CPU and GPU acceleration speedup. Full integer quantization offers the smallest model size, but it is 4x larger than float16 quantization and 2x larger than dynamic range quantization. The best hardware choice for each quantization method would be determined by the specific hardware capabilities and the trade-offs between size and performance.

> TFLite 的 dynamic range 量化帶來最小的模型大小，但代價是延遲適度增加。它最適合能處理較高延遲的 CPU 與硬體加速器。Float16 量化提供最小的模型大小，但代價是 CPU 與 GPU 加速的加速倍數增加 2 倍。Full integer 量化提供最小的模型大小，但它比 float16 量化大 4 倍、比 dynamic range 量化大 2 倍。每種量化方法的最佳硬體選擇，取決於具體的硬體能力以及大小與效能之間的取捨。
