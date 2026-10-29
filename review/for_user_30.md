# D9 覆核：請對每個答案打 0／1／2 分

- **2**：與標準答案的關鍵事實都一致，沒有錯誤內容
- **1**：對了一部分（缺關鍵點，或夾帶小錯）
- **0**：錯誤、離題、答案被截斷，或有答案題卻拒答
- 無答案題（標準答案為 Not stated）：正確拒答＝2；拒答但編造縮寫全名＝1；直接作答＝0

回覆格式範例：`1=2, 2=0, 3=1, ...`（30 題）

## 1. [fact] Where is PyTorch centralizing its quantization development?
**標準答案**：In the torchao project (pytorch/ao); the old torch.ao.quantization flows are being migrated there.

**模型答案**：

> PyTorch is centralizing its quantization development through the introduction of the PyTorch 2 Export Quantization flow, which includes support for both static and dynamic quantization, and utilizes the torch.export mechanism to capture the model into a graph. This flow also integrates with the TorchInductor compiler backend, allowing for the generation of a quantized model customized for the x86 inductor backend. The flow supports programmability and simplifies the user experience by providing a single instance for both backend and user interaction.

## 2. [compare] What data does each TFLite quantization technique need: float16, dynamic range, integer, and quantization-aware training?
**標準答案**：Float16 and dynamic range need no data; integer quantization needs an unlabelled representative sample; quantization-aware training needs labelled training data.

**模型答案**：

> For float16 quantization, the model weights are quantized to float16. For dynamic range quantization, activations are dynamically quantized to 8 bits during inference. For integer only quantization, the model weights are quantized to 8-bit integers and activations to 16-bit integers. Quantization-aware training requires retraining the model to compensate for quantization-induced errors.

## 3. [compare] How should users migrate from PyTorch's eager mode and FX graph mode quantization flows?
**標準答案**：Eager mode quantization users should migrate to torchao's quantize_ API; FX graph mode users should migrate to torchao's pt2e quantization API (prepare_pt2e and convert_pt2e). pt2e quantization itself has moved to torchao.

**模型答案**：

> To migrate from PyTorch's eager mode and FX graph mode quantization flows, users should follow the tutorial on PyTorch 2 Export Quantization. This tutorial explains the steps to generate a quantized model using the PyTorch 2 Export Quantization flow, which is designed for the x86 Inductor backend. The tutorial provides a detailed workflow that includes capturing the model, applying quantization, and lowering the model into the Inductor. Users are advised to implement a `Quantizer` for their specific backend, as the model produced at this stage is not the final model that runs on the device. The tutorial also provides guidance on debugging and performance evaluation of the quantized models. Therefore, users should migrate to the PyTorch 2 Export Quantization flow as a more advanced and flexible approach for quantization.

## 4. [unanswerable] How do I run a GGUF Q4_K_M model with llama.cpp?
**標準答案**：Not stated in the documents.

**模型答案**：

> To run a GGUF Q4_K_M model with `llama.cpp` on a mobile device, you would need to follow the instructions provided for iOS devices. According to the context, this can be done using the `.pte` file generated from the model. The instructions for iOS devices are detailed in the provided link. Therefore, you would need to follow the iOS device instructions to run the model.

## 5. [fact] Which TFLite post-training quantization method is recommended as a starting point, and why?
**標準答案**：Dynamic range quantization, because it reduces memory usage and speeds up computation without requiring a representative dataset for calibration.

**模型答案**：

> Dynamic range quantization is recommended as a starting point because it reduces memory usage and speeds up computation, providing 8-bit precision with only the weights quantized at conversion time, and it is a recommended approach for reduced memory usage and faster computation. However, it still requires floating-point storage for the outputs.

## 6. [compare] How do dynamic quantization and weight-only quantization differ in torchao's quantized inference?
**標準答案**：Dynamic quantization converts both the input activations and the weights to low precision (for example fp8 rowwise) before the matmul. Weight-only quantization (for example int4) quantizes only the weight and keeps the activations in high precision.

**模型答案**：

> In torchao's quantized inference, dynamic quantization involves converting the input and weight tensors to a quantized format (like fp8) before performing the matrix multiplication. Weight-only quantization, on the other hand, only converts the weight tensor to a quantized format (like int4) while keeping the input tensor in its original precision.

## 7. [compare] What data does each TFLite quantization technique need: float16, dynamic range, integer, and quantization-aware training?
**標準答案**：Float16 and dynamic range need no data; integer quantization needs an unlabelled representative sample; quantization-aware training needs labelled training data.

**模型答案**：

> For Post-training float16 quantization, no data is needed. For Post-training dynamic range quantization, no data is also needed. For Post-training integer quantization, an unlabelled representative sample is required. For quantization-aware training, labelled training data is needed.

## 8. [number] In the TFLite tutorial, how much does dynamic range quantization shrink the ResNet v2 101 model?
**標準答案**：From 171 MB to 43 MB, with the same top-1 accuracy of 76.8.

**模型答案**：

> I cannot find this in the provided documents.

## 9. [fact] What formula does ONNX Runtime use to map a quantized value back to a floating point value?
**標準答案**：val_fp32 = scale * (val_quantized - zero_point).

**模型答案**：

> ONNX Runtime uses the formula provided in the context to map a quantized value back to a floating point value. The formula is:
> 
> ```
> val_fp32 = scale * (val_quantized - zero_point)
> ``
> 
> Where `scale` and `zero_point` are calculated based on the data range and quantization range, as detailed in the context. The inverse operation is not directly provided, but the original floating point value can be derived from the quantized value using this inverse formula.

## 10. [fact] What known limitation does ONNX Runtime have when optimizing large models during quantization pre-processing?
**標準答案**：Model optimization cannot output a model larger than 2GB, so it must be skipped for large models.

**模型答案**：

> ONNX Runtime has a known limitation where model optimization during quantization pre-processing can only output a model size greater than 2GB. For large models, optimization must be skipped.

## 11. [fact] What is the prerequisite for using the PT2E post-training quantization flow?
**標準答案**：The model must be exportable with torch.export.export.

**模型答案**：

> Exportable by `torch.export.export` is a prerequisite for using the PT2E post-training quantization flow.

## 12. [compare] How do dynamic quantization and weight-only quantization differ in torchao's quantized inference?
**標準答案**：Dynamic quantization converts both the input activations and the weights to low precision (for example fp8 rowwise) before the matmul. Weight-only quantization (for example int4) quantizes only the weight and keeps the activations in high precision.

**模型答案**：

> Dynamic quantization and weight-only quantization in torchao's quantized inference differ in how they handle the weights. In dynamic quantization, the weights are converted to a lower precision format (like fp8 in the example), while in weight-only quantization, only the weights are quantized to a lower precision (like int4 in the example), leaving the inputs at their full precision.

## 13. [fact] In the PT2E QAT flow, why should you not call model.eval() or model.train() after program capture?
**標準答案**：These methods no longer correctly change the behavior of ops like dropout and batch normalization after export; use move_exported_model_to_eval (and move_exported_model_to_train) instead.

**模型答案**：

> In the PyTorch 2 Export Quantization-Aware Training (QAT) flow, you should not call `model.eval()` or `model.train()` after program capture because these methods no longer correctly change the behavior of certain operations like dropout and batch normalization. Instead, you must use the provided functions to ensure these operations behave correctly in the evaluation or training modes, respectively. This is to maintain the integrity of the operations' behavior during inference.

## 14. [fact] In ONNX Runtime quantization, why must the floating point zero be exactly representable in the quantization space?
**標準答案**：Because zero padding is used in many CNNs. If 0 cannot be represented uniquely after quantization, it causes accuracy errors.

**模型答案**：

> In ONNX Runtime quantization, the floating point zero must be exactly representable in the quantization space to avoid accuracy errors. If the floating point zero value cannot be uniquely represented after quantization, it can lead to inaccuracies in the model's performance.

## 15. [compare] What is the difference between TFLite integer quantization with float fallback and integer-only quantization?
**標準答案**：With float fallback, ops without an integer implementation stay in float and the input and output stay float, so the model is not compatible with integer-only devices such as 8-bit microcontrollers or Coral Edge TPU. Integer-only enforces integer quantization for all ops including input and output, for those devices.

**模型答案**：

> The difference between TFLite integer quantization with float fallback and integer-only quantization lies in the compatibility with devices and accelerators. Integer with float fallback allows the use of float operators when integer implementations are not available, making it smoother for conversion but not compatible with devices that require integer inputs and outputs. In contrast, integer-only quantization ensures that the model is fully quantized to integers, including the input and output, making it compatible with devices like 8-bit microcontrollers and Coral Edge TPUs.

## 16. [number] What CER does DeepSpeech 0.5.1 reach with float32, int8 and int16 activations in the TFLite int16 table?
**標準答案**：6.13% with float32 activations, 43.67% with int8 activations, and 6.52% with int16 activations.

**模型答案**：

> The CER (Character Error Rate) for DeepSpeech 0.5.1 with int8 activations is 43.67%. For int16 activations, it is 6.52%. The table does not provide a float32 activation CER for DeepSpeech 0.5.1.

## 17. [fact] What are the two steps of QAT in torchao, and what does each do?
**標準答案**：Prepare inserts fake quantize operations into linear layers; convert turns the fake quantize operations into actual quantize and dequantize operations after training.

**模型答案**：

> In torchao, the two steps of Quantization-Aware Training (QAT) are:
> 
> 1. **Prepare**: This step inserts fake quantize operations into linear layers of the model. This allows the model to be trained as if it were quantized, but without actually changing the data types of the weights and activations.
> 
> 2. **Convert**: After training, this step transforms the fake quantize operations into actual quantize and dequantize operations. This step is crucial for producing a quantized model, as it replaces the fake quantize operations with the actual quantization and dequantization that would be used in a post-training quantization process.

## 18. [compare] How does float16 conversion in TFLite compare with float16 conversion in ONNX Runtime?
**標準答案**：Both can shrink the model by up to half with small accuracy loss and benefit GPUs. TFLite float16 quantizes weights and dequantizes them to float32 on CPU by default, while the GPU delegate runs float16 directly. ONNX Runtime converts the model with a float16 tool, needs no tuning data, may need op block lists or mixed precision, and the CPU version does not support float16 ops.

**模型答案**：

> The provided context does not contain information that directly compares the float16 conversion process in TFLite to that in ONNX Runtime. The information given is specific to the float16 quantization process in TensorFlow and ONNX, and does not provide a direct comparison between the two. Therefore, I cannot find this information in the provided documents.

## 19. [unanswerable] What is the latency of an int8 MobileNetV2 in TFLite on a Raspberry Pi 4?
**標準答案**：Not stated in the documents.

**模型答案**：

> The provided context does not contain information about the latency of an int8 MobileNetV2 in TFLite on a Raspberry Pi 4. The details about the latency of MobileNetV2 on a Raspberry Pi 4, particularly in the context of int8 (intelligent) quantization, are not included in the given information. Therefore, I cannot find this information in the provided documents.

## 20. [fact] Why does TFLite full integer quantization require a representative dataset?
**標準答案**：Variable tensors such as inputs, activations and outputs cannot be calibrated unless inference is run, so the converter needs sample data to estimate their (min, max) ranges.

**模型答案**：

> TFLite full integer quantization requires a representative dataset because variable tensors such as model input, activations, and model output cannot be calibrated with just the model's weights and biases. These tensors need to be calibrated through actual inference cycles, which necessitates a dataset to simulate this process. The representative dataset provides the necessary data points to estimate the range of these tensors, allowing for proper quantization.

## 21. [number] What CER does DeepSpeech 0.5.1 reach with float32, int8 and int16 activations in the TFLite int16 table?
**標準答案**：6.13% with float32 activations, 43.67% with int8 activations, and 6.52% with int16 activations.

**模型答案**：

> I cannot find this in the provided documents. The CER (Categorical Cross-Entropy) values for DeepSpeech 0.5.1 with float32, int8, and int16 activations in the TFLite int16 table are not specified in the provided context.

## 22. [fact] What known limitation does ONNX Runtime have when optimizing large models during quantization pre-processing?
**標準答案**：Model optimization cannot output a model larger than 2GB, so it must be skipped for large models.

**模型答案**：

> The known limitation of ONNX Runtime when optimizing large models during quantization pre-processing is that it cannot output a model size greater than 2GB. So for large models, optimization must be skipped. This is detailed in the provided context.

## 23. [compare] How do the PT2E post-training quantization and PT2E QAT flows differ?
**標準答案**：Both capture the model with torch.export and end with convert_pt2e. PTQ uses prepare_pt2e followed by calibration, while QAT uses prepare_qat_pt2e followed by training, and after export QAT must use move_exported_model_to_eval instead of model.eval().

**模型答案**：

> The PT2E post-training quantization and PT2E QAT flows differ in the steps of quantization-aware training (QAT). In QAT, the quantization-aware training step involves using a custom quantizer to prepare the model for quantization, which allows for the model to be trained with quantization in mind. This step is not present in the standard PT2E flow, which directly quantizes the model after capturing the model's operations.

## 24. [compare] What is the difference between post-training quantization and quantization-aware training?
**標準答案**：PTQ quantizes an already-trained model without retraining and may lose accuracy, especially on small networks. QAT applies fake quantization during training or fine-tuning so the final quantized model has higher accuracy, but needs changes to the training process.

**模型答案**：

> Post-training quantization and quantization-aware training (QAT) are both techniques used to optimize models for reduced size and improved performance, but they approach the process differently. Post-training quantization is a one-time process that converts an already-trained model into a quantized version, which can reduce the model size and improve performance, but it does not require any changes to the training process. In contrast, quantization-aware training involves modifying the training process to include quantization operations, which can lead to better accuracy and lower perplexities, but it requires a more complex and time-consuming training process. The decision between these two methods depends on the specific requirements of the application, such as the desired balance between model size, accuracy, and performance.

## 25. [compare] What is the difference between the QOperator and QDQ formats for quantized ONNX models?
**標準答案**：QOperator uses dedicated quantized operators such as QLinearConv and MatMulInteger. QDQ inserts DeQuantizeLinear(QuantizeLinear(tensor)) between the original operators to simulate quantization and dequantization.

**模型答案**：

> The QOperator format represents quantized ONNX models where each quantized operator has its own ONNX definition, such as QLinearConv, MatMulInteger, etc. In contrast, the QDQ format inserts DeQuantizeLinear(QuantizeLinear(tensor)) between the original operators to simulate the quantization and dequantization process. In static quantization, the QuantizeLinear and DeQuantizeLinear operators carry quantization parameters, while in dynamic quantization, a ComputeQuantizationParameters function proto is used to calculate these parameters on the fly.

## 26. [unanswerable] How does the AWQ algorithm decide which weight channels to protect from quantization error?
**標準答案**：Not stated in the documents.

**模型答案**：

> The provided context does not explicitly detail how the AWQ (Weight Quantization) algorithm decides which weight channels to protect from quantization error. The information about the AWQ algorithm and its decision-making process for protecting specific weight channels is not provided in the given context. Therefore, I cannot find this information in the provided documents.

## 27. [fact] What is the prerequisite for using the PT2E post-training quantization flow?
**標準答案**：The model must be exportable with torch.export.export.

**模型答案**：

> The prerequisite for using the PT2E post-training quantization flow is having TensorFlow 1.15 or higher. This is mentioned in the context provided in note [1].

## 28. [compare] How do dynamic and static quantization differ in ONNX Runtime?
**標準答案**：Dynamic quantization computes activation scale and zero point at inference time, which adds cost but usually gives higher accuracy. Static quantization runs calibration data beforehand and stores the activation parameters as constants in the model.

**模型答案**：

> Dynamic quantization calculates quantization parameters (scale and zero point) for activations during the quantization process, which increases inference cost but usually achieves higher accuracy compared to static quantization. Static quantization, on the other hand, involves running the model with calibration data to compute these parameters, which are then written as constants in the quantized model and used for all inputs.

## 29. [number] What F1 does MobileBert get with float32 activations versus int8 activations in the TFLite int16 table?
**標準答案**：F1 88.81 with float32 activations and 2.08 with int8 activations (88.73 with int16 activations).

**模型答案**：

> The table provided does not contain the specific F1 (Exact match) F1(Exact match) accuracy values for MobileBert with float32 activations versus int8 activations in the TFLite int16 table. The provided data only includes the F1 accuracy for MobileBert with int8 activations, not the comparison with float32 activations. Therefore, I cannot find this information in the provided documents.

## 30. [compare] How do TFLite dynamic range, full integer and float16 quantization compare in size reduction and hardware?
**標準答案**：Dynamic range: 4x smaller, 2x-3x speedup, CPU. Full integer: 4x smaller, 3x+ speedup, CPU, Edge TPU and microcontrollers. Float16: 2x smaller with GPU acceleration, CPU and GPU.

**模型答案**：

> TFLite dynamic range quantization results in the smallest model size, though at the expense of a moderate increase in latency. It is best for CPU and hardware accelerators that can handle the increased latency. The float16 quantization provides the smallest model size, though at the cost of a 2x increase in CPU and GPU acceleration speedup. Full integer quantization offers the smallest model size, but it is 4x larger than float16 quantization and 2x larger than dynamic range quantization. The best hardware choice for each quantization method would be determined by the specific hardware capabilities and the trade-offs between size and performance.
