# 依據抽查（10 題）

**怎麼判斷**：只看「答案裡的事實」能不能在該題的 Context 找到。

- 有依據（1）：答案的每個事實都出自 Context（即使跟標準答案不同也算）
- 沒依據（0）：答案含有 Context 沒有的內容（編造、外部知識、推論過頭）

**不用管答案對不對**，只看有沒有憑空多出來的東西。Context 為英文原文，未翻譯。

---

## 1

**Question**

How do TFLite dynamic range, full integer and float16 quantization compare in size reduction and hardware?

> TFLite 的 dynamic range、full integer、float16 量化，在模型縮小幅度與硬體上如何比較？

**Answer**

TFLite dynamic range quantization reduces the model size by 4x and provides a 2x-3x speedup, suitable for CPU. Full integer quantization achieves a 4x reduction in size and a 3x+ speedup, supporting CPU, Edge TPU, and microcontrollers. Float16 quantization results in a 2x smaller model with GPU acceleration, offering a 2x-3x speedup for GPU.

> TFLite dynamic range 量化讓模型縮小 4 倍、加速 2x-3x，適用 CPU。Full integer 量化縮小 4 倍、加速 3x 以上，支援 CPU、Edge TPU 與 microcontroller。Float16 量化讓模型縮小 2 倍並可用 GPU 加速，在 GPU 上提供 2x-3x 加速。

**Context**（1 段）

`tflite__post_training_quantization#000`

````
Post-training quantization

Post-training quantization is a conversion technique that can reduce model size
while also improving CPU and hardware accelerator latency, with little
degradation in model accuracy. You can quantize an already-trained float
TensorFlow model when you convert it to TensorFlow Lite format using the
[TensorFlow Lite Converter](../models/convert/).

Note: The procedures on this page require TensorFlow 1.15 or higher.

Optimization Methods

There are several post-training quantization options to choose from. Here is a
summary table of the choices and the benefits they provide:

| Technique            | Benefits                  | Hardware         |
| -------------------- | ------------------------- | ---------------- |
| Dynamic range        | 4x smaller, 2x-3x speedup | CPU              |
: quantization         :                           :                  :
| Full integer         | 4x smaller, 3x+ speedup   | CPU, Edge TPU,   |
: quantization         :                           : Microcontrollers :
| Float16 quantization | 2x smaller, GPU           | CPU, GPU         |
:                      : acceleration              :                  :

The following decision tree can help determine which post-training quantization
method is best for your use case:

![post-training optimization options](images/optimization.jpg)
````

---

## 2

**Question**

What formula does ONNX Runtime use to map a quantized value back to a floating point value?

> ONNX Runtime 用什麼公式把量化值映射回浮點數？

**Answer**

ONNX Runtime uses the formula provided in the context to map a quantized value back to a floating point value. The formula is:

```
val_fp32 = scale * (val_quantized - zero_point)
``

Where `scale` and `zero_point` are calculated based on the data range and quantization range, as detailed in the context. The inverse operation is not directly provided, but the original floating point value can be derived from the quantized value using this inverse formula.
```

> ONNX Runtime 使用 context 中提供的公式把量化值映射回浮點數。公式為：val_fp32 = scale * (val_quantized - zero_point)。其中 scale 與 zero_point 依資料範圍與量化範圍計算，如 context 所述。反向運算並未直接提供，但可用此反向公式由量化值推得原始浮點值。

**Context**（1 段）

`ort__quantization#000`

````
Quantization Overview

Quantization in ONNX Runtime refers to 8 bit linear quantization of an ONNX model.

During quantization, the floating point values are mapped to an 8 bit quantization space of the form:
 `val_fp32 = scale * (val_quantized - zero_point)`

`scale` is a positive real number used to map the floating point numbers to a quantization space. It is calculated as follows:

For asymmetric quantization:
 ```
 scale = (data_range_max - data_range_min) / (quantization_range_max - quantization_range_min)
 ```

For symmetric quantization:
 ```
 scale = max(abs(data_range_max), abs(data_range_min)) * 2 / (quantization_range_max - quantization_range_min)
 ```

`zero_point` represents zero in the quantization space. It is important that the floating point zero value be exactly representable in quantization space. This is because zero padding is used in many CNNs. If it is not possible to represent 0 uniquely after quantization, it will result in accuracy errors.
````

---

## 3

**Question**

Why does the TFLite 8-bit quantization spec force weights to be symmetric with zero-point equal to 0?

> 為什麼 TFLite 8-bit 量化規格強制 weights 為對稱且 zero-point 等於 0？

**Answer**

The TFLite 8-bit quantization spec forces weights to be symmetric with a zero-point equal to 0 to avoid an unavoidable runtime cost of multiplying the zero-point of the weight with the activation value. This ensures that the quantization is as efficient as possible, avoiding unnecessary computational overhead. Symmetric weights allow for the optimization of the zero-point value, which is an inexpensive way to achieve additional precision.

> TFLite 8-bit 量化規格強制 weights 對稱且 zero-point 為 0，是為了避免 weight 的 zero-point 與 activation 值相乘所造成的無法避免的 runtime 成本。這讓量化盡可能有效率、避免不必要的運算負擔。對稱的 weights 讓 zero-point 值得以最佳化，這是取得額外精度的一種低成本方式。

**Context**（1 段）

`tflite__quantization_spec#003`

````
Symmetric vs asymmetric

Activations are asymmetric: they can have their zero-point anywhere within the
signed `int8` range `[-128, 127]`. Many activations are asymmetric in nature and
a zero-point is an relatively inexpensive way to effectively get up to an extra
binary bit of precision. Since activations are only multiplied by constant
weights, the constant zero-point value can be optimized pretty heavily.

Weights are symmetric: forced to have zero-point equal to 0. Weight values are
multiplied by dynamic input and activation values. This means that there is an
unavoidable runtime cost of multiplying the zero-point of the weight with the
activation value. By enforcing that zero-point is 0 we can avoid this cost.

Explanation of the math: this is similar to section 2.3 in
[arXiv:1712.05877](https://arxiv.org/abs/1712.05877), except for the difference
that we allow the scale values to be per-axis. This generalizes readily, as
follows:

$A$ is a $m \times n$ matrix of quantized activations. <br />
$B$ is a $n \times p$ matrix of quantized weights. <br />
Consider multiplying the $j$th row of $A$, $a_j$ by the $k$th column of
$B$, $b_k$, both of length $n$. The quantized integer values and
zero-points values are $q_a$, $z_a$ and $q_b$, $z_b$ respectively.
````

---

## 4

**Question**

Why does the PyTorch 2 Export (PT2E) quantization flow claim to be better than FX Graph Mode Quantization?

> 為什麼 PyTorch 2 Export (PT2E) 量化流程宣稱比 FX Graph Mode Quantization 更好？

**Answer**

The PyTorch 2 Export (PT2E) quantization flow is claimed to be better than the FX Graph Mode Quantization because it offers significantly higher model coverage, better programmability, and a simpler user experience. This is indicated by the statement that this flow is expected to have significantly higher model coverage, better programmability, and a simplified user experience compared to the FX Graph Mode Quantization. The tutorial also mentions that the exportable constructs are significantly more extensive for the PT2E flow, which is a key factor in achieving higher coverage.

> PT2E 量化流程被宣稱比 FX Graph Mode Quantization 更好，因為它提供明顯更高的模型覆蓋率、更好的可程式化能力，以及更簡化的使用體驗。這來自於「預期此流程比 FX Graph Mode Quantization 有明顯更高的模型覆蓋率、更好的可程式化能力與更簡化的使用體驗」這段敘述。教學也提到 PT2E 流程可 export 的 construct 明顯更多，這是達到更高覆蓋率的關鍵因素。

**Context**（1 段）

`pytorch__pt2e_quant_ptq#000`

````
PyTorch 2 Export Post Training Quantization

**Author**: [Jerry Zhang](https://github.com/jerryzh168)

This tutorial introduces the steps to do post training static
quantization in graph mode based on
[torch.\_export.export](https://pytorch.org/docs/main/export.html).
Compared to [FX Graph Mode
Quantization](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html),
this flow is expected to have significantly higher model coverage ([88%
on 14K
models](https://github.com/pytorch/pytorch/issues/93667#issuecomment-1601171596)),
better programmability, and a simplified UX.

Exportable by <span class="title-ref">torch.export.export</span> is a
prerequisite to use the flow, you can find what are the constructs
that's supported in [Export
DB](https://pytorch.org/docs/main/generated/exportdb/index.html).

The high level architecture of quantization 2 with quantizer could look
like this:
````

---

## 5

**Question**

What are the two steps of QAT in torchao, and what does each do?

> torchao 的 QAT 有哪兩個步驟？各自做什麼？

**Answer**

In Quantization-Aware Training (QAT) within torchao, the two steps are:

1. **Prepare**: This step inserts fake quantization operations into linear layers. This involves modifying the model to include these operations that simulate the behavior of quantization and dequantization during the training phase.

2. **Convert**: This step transforms the simulated quantization operations into actual quantization and dequantization operations after the training phase. This involves a final step where the dequantization operations are fused with the linear layer, and the model is then ready for quantization.

Each of these steps is crucial for preparing the model to handle quantization during and after training, ensuring the final quantized model can achieve higher accuracies and lower perplexities.

> 在 torchao 的 Quantization-Aware Training (QAT) 中，兩個步驟是：
>
> 1. **Prepare**：在 linear layer 插入 fake quantization 運算。這會修改模型，加入在訓練階段模擬 quantization 與 dequantization 行為的運算。
>
> 2. **Convert**：訓練後把模擬的 quantization 運算轉成真正的 quantization 與 dequantization 運算。最後一步會把 dequantization 運算與 linear layer 融合，模型即可進行量化。
>
> 這兩步都是讓模型在訓練中與訓練後能處理量化的關鍵，確保最終量化模型能達到更高準確度與更低 perplexity。

**Context**（1 段）

`pytorch__qat#000`

````
Quantization-Aware Training (QAT)

Quantization-Aware Training (QAT) refers to applying fake quantization during the
training or fine-tuning process, such that the final quantized model will exhibit
higher accuracies and lower perplexities. Fake quantization refers to rounding the float
values to quantized values without actually casting them to dtypes with lower
bit-widths, in contrast to post-training quantization (PTQ), which does cast the
quantized values to lower bit-width dtypes, e.g.:

```
# PTQ: x_q is quantized and cast to int8
# scale and zero point (zp) refer to parameters used to quantize x_float
# qmin and qmax refer to the range of quantized values
x_q = (x_float / scale + zp).round().clamp(qmin, qmax).cast(int8)

# QAT: x_fq is still in float
# Fake quantize simulates the numerics of quantize + dequantize
x_fq = (x_float / scale + zp).round().clamp(qmin, qmax)
x_fq = (x_fq - zp) * scale
```

QAT typically involves applying a transformation to your model before and after training.
In torchao, these are represented as the prepare and convert steps: (1) prepare inserts
fake quantize operations into linear layers, and (2) convert transforms the fake quantize
operations to actual quantize and dequantize operations after training, thereby producing
a quantized model (dequantize operations are typically fused with linear after lowering).
Between these two steps, training can proceed exactly as before.
````

---

## 6

**Question**

Why does the TFLite 8-bit quantization spec force weights to be symmetric with zero-point equal to 0?

> 為什麼 TFLite 8-bit 量化規格強制 weights 為對稱且 zero-point 等於 0？

**Answer**

The TFLite 8-bit quantization spec forces weights to be symmetric with a zero-point equal to 0 to avoid the cost of multiplying the zero-point of the weight with the activation value. This optimization is necessary for the math in the quantization formula, ensuring that the term \(\sum_{i=0}^{n} q_{a}^{(i)} z_{b}\) is computed only once, thus removing an unnecessary runtime cost. This optimization is crucial for maintaining the performance of the quantized model, especially in scenarios where the cost of computing this term is significant.

> TFLite 8-bit 量化規格強制 weights 對稱且 zero-point 為 0，是為了避免 weight 的 zero-point 與 activation 值相乘的成本。這項最佳化對量化公式的數學是必要的，使 \(\sum_{i=0}^{n} q_{a}^{(i)} z_{b}\) 這一項只需計算一次，因而去除不必要的 runtime 成本。這項最佳化對維持量化模型的效能很重要，尤其是在計算這一項成本很高的情境。

**Context**（4 段）

`tflite__quantization_spec#003`

````
Symmetric vs asymmetric

Activations are asymmetric: they can have their zero-point anywhere within the
signed `int8` range `[-128, 127]`. Many activations are asymmetric in nature and
a zero-point is an relatively inexpensive way to effectively get up to an extra
binary bit of precision. Since activations are only multiplied by constant
weights, the constant zero-point value can be optimized pretty heavily.

Weights are symmetric: forced to have zero-point equal to 0. Weight values are
multiplied by dynamic input and activation values. This means that there is an
unavoidable runtime cost of multiplying the zero-point of the weight with the
activation value. By enforcing that zero-point is 0 we can avoid this cost.

Explanation of the math: this is similar to section 2.3 in
[arXiv:1712.05877](https://arxiv.org/abs/1712.05877), except for the difference
that we allow the scale values to be per-axis. This generalizes readily, as
follows:

$A$ is a $m \times n$ matrix of quantized activations. <br />
$B$ is a $n \times p$ matrix of quantized weights. <br />
Consider multiplying the $j$th row of $A$, $a_j$ by the $k$th column of
$B$, $b_k$, both of length $n$. The quantized integer values and
zero-points values are $q_a$, $z_a$ and $q_b$, $z_b$ respectively.
````

`tflite__quantization_spec#005`

````
The \\(\sum_{i=0}^{n} q_{a}^{(i)} z_b\\) term needs to be computed every inference
since the activation changes every inference. By enforcing weights to be
symmetric we can remove the cost of this term.
````

`tflite__post_training_quantization#010`

````
Representation for quantized tensors

8-bit quantization approximates floating point values using the following
formula.

$$real\_value = (int8\_value - zero\_point) \times scale$$

The representation has two main parts:

*   Per-axis (aka per-channel) or per-tensor weights represented by int8 two’s
    complement values in the range [-127, 127] with zero-point equal to 0.

*   Per-tensor activations/inputs represented by int8 two’s complement values in
    the range [-128, 127], with a zero-point in range [-128, 127].

For a detailed view of our quantization scheme, please see our
[quantization spec](./quantization_spec). Hardware vendors who want to plug
into TensorFlow Lite's delegate interface are encouraged to implement the
quantization scheme described there.
````

`tflite__post_training_quantization#008`

````
If 16x8 quantization is not supported for some operators in the model,
then the model still can be quantized, but unsupported operators kept in float.
The following option should be added to the target_spec to allow this.
<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.representative_dataset = representative_dataset
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
<b>tf.lite.OpsSet.TFLITE_BUILTINS</b>]
tflite_quant_model = converter.convert()
</pre>

Examples of the use cases where accuracy improvements provided by this
quantization scheme include:

*   super-resolution,
*   audio signal processing such as noise cancelling and beamforming,
*   image de-noising,
*   HDR reconstruction from a single image.

The disadvantage of this quantization is:

*   Currently inference is noticeably slower than 8-bit full integer due to the
    lack of optimized kernel implementation.
*   Currently it is incompatible with the existing hardware accelerated TFLite
    delegates.

Note: This is an experimental feature.

A tutorial for this quantization mode can be found
[here](post_training_integer_quant_16x8.ipynb).
````

---

## 7

**Question**

How do dynamic and static quantization differ in ONNX Runtime?

> ONNX Runtime 中 dynamic 與 static quantization 有何不同？

**Answer**

Dynamic quantization calculates quantization parameters (scale and zero point) for activations during the quantization process, which increases inference cost but usually achieves higher accuracy compared to static quantization. Static quantization, on the other hand, involves running the model with calibration data to compute these parameters, which are then written as constants in the quantized model and used for all inputs.

> Dynamic quantization 在量化過程中計算 activation 的量化參數（scale 與 zero point），這會增加 inference 成本，但通常比 static quantization 準確度更高。Static quantization 則是用 calibration data 跑模型來計算這些參數，再把它們寫成量化模型中的常數，供所有輸入使用。

**Context**（1 段）

`ort__quantization#006`

````
Dynamic Quantization

There are two ways of quantizing a model: dynamic and static. Dynamic quantization calculates the
quantization parameters (scale and zero point) for activations dynamically. These calculations
increase the cost of inference, while usually achieve higher accuracy comparing to static ones.

Python API for dynamic quantization is in module `onnxruntime.quantization.quantize`, function `quantize_dynamic()`

Static Quantization

Static quantization method first runs the model using a set of inputs called calibration data.
During these runs, we compute the quantization parameters for each activations. These quantization
parameters are written as constants to the quantized model and used for all inputs. Our
quantization tool supports three calibration methods: MinMax, Entropy and Percentile. Please refer
to
[`calibrate.py`](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/calibrate.py)
for details.

Python API for static quantization is in module `onnxruntime.quantization.quantize`, function
`quantize_static()`.
Please refer to [quantize.py](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/quantize.py)
for details.
````

---

## 8

**Question**

How do per-tensor and per-axis quantization differ in the TFLite quantization spec?

> 在 TFLite 量化規格中，per-tensor 與 per-axis 量化有何不同？

**Answer**

Per-tensor quantization applies a single scale and zero-point value to the entire tensor, while per-axis quantization applies these values on a per-axis basis, specifically for slices in the `quantized_dimension`. In per-tensor quantization, the same scale and zero-point values are used across the entire tensor, whereas in per-axis quantization, different values can be applied for each axis, particularly for the specified dimension. This allows for more granular quantization in TFLite, improving accuracy in specific dimensions or layers.

> Per-tensor 量化對整個 tensor 套用單一的 scale 與 zero-point，而 per-axis 量化是依每個 axis 套用這些值，具體是針對 `quantized_dimension` 中的 slice。Per-tensor 在整個 tensor 使用相同的 scale 與 zero-point，per-axis 則可對每個 axis（特別是指定的維度）套用不同的值。這讓 TFLite 的量化更細緻，能提升特定維度或 layer 的準確度。

**Context**（1 段）

`tflite__quantization_spec#002`

````
Per-axis vs per-tensor

Per-tensor quantization means that there will be one scale and/or zero-point per
entire tensor. Per-axis quantization means that there will be one scale and/or
`zero_point` per slice in the `quantized_dimension`. The quantized dimension
specifies the dimension of the Tensor's shape that the scales and zero-points
correspond to. For example, a tensor `t`, with `dims=[4, 3, 2, 1]` with
quantization params: `scale=[1.0, 2.0, 3.0]`, `zero_point=[1, 2, 3]`,
`quantization_dimension=1` will be quantized across the second dimension of `t`:

t[:, 0, :, :] will have scale[0]=1.0, zero_point[0]=1
    t[:, 1, :, :] will have scale[1]=2.0, zero_point[1]=2
    t[:, 2, :, :] will have scale[2]=3.0, zero_point[2]=3

Often, the `quantized_dimension` is the `output_channel` of the weights of
convolutions, but in theory it can be the dimension that corresponds to each
dot-product in the kernel implementation, allowing more quantization granularity
without performance implications. This has large improvements to accuracy.

TFLite has per-axis support for a growing number of operations. At the time of
this document, support exists for Conv2d and DepthwiseConv2d.
````

---

## 9

**Question**

Which TFLite post-training quantization method is recommended as a starting point, and why?

> TFLite 的 post-training quantization 方法中，哪一個建議作為起點？為什麼？

**Answer**

The method recommended as a starting point for post-training quantization is **Dynamic range quantization**. This is because it achieves a 4x reduction in the model size and a 2x-3x speedup, which are significant benefits. Additionally, it supports on-the-fly quantization and dequantization, allowing for mixing of quantized and floating-point kernels, which can improve performance for different parts of the graph. However, it is important to ensure the accuracy of the quantized model, as this method does not retrain the model to compensate for the quantization. Therefore, it is a good starting point for evaluating the benefits of quantization.

> 建議作為 post-training quantization 起點的方法是 **Dynamic range quantization**。因為它能讓模型縮小 4 倍並加速 2x-3x，效益顯著。此外，它支援 on-the-fly 的 quantization 與 dequantization，可混合 quantized 與 floating-point kernel，對圖中不同部分提升效能。不過要注意確認量化模型的準確度，因為此方法不會重新訓練模型來補償量化。因此它適合作為評估量化效益的起點。

**Context**（4 段）

`tflite__post_training_integer_quant#013`

````
First, we need a function that runs inference with a given model and images, and then returns the predictions:

```python
# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_image_indices):
  global test_images

  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(test_image_indices),), dtype=int)
  for i, test_image_index in enumerate(test_image_indices):
    test_image = test_images[test_image_index]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details["quantization"]
      test_image = test_image / input_scale + input_zero_point

    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    predictions[i] = output.argmax()

  return predictions
```
````

`tflite__post_training_quant#002`

````
Overview

[TensorFlow Lite](https://www.tensorflow.org/lite/) now supports
converting weights to 8-bit precision as part of model conversion from
tensorflow graphdefs to TensorFlow Lite's flat buffer format. Dynamic range quantization achieves a 4x reduction in the model size. In addition, TFLite supports on-the-fly quantization and dequantization of activations to allow for:

1.  Using quantized kernels for faster implementation when available.
2.  Mixing of floating-point kernels with quantized kernels for different parts
    of the graph.

The activations are always stored in floating point. For ops that
support quantized kernels, the activations are quantized to 8 bits of precision
dynamically prior to processing and are de-quantized to float precision after
processing. Depending on the model being converted, this can give a speedup over
pure floating point computation.

In contrast to
[quantization aware training](https://github.com/tensorflow/tensorflow/tree/r1.14/tensorflow/contrib/quantize)
, the weights are quantized post training and the activations are quantized dynamically 
at inference in this method.
Therefore, the model weights are not retrained to compensate for quantization
induced errors. It is important to check the accuracy of the quantized model to
ensure that the degradation is acceptable.

This tutorial trains an MNIST model from scratch, checks its accuracy in
TensorFlow, and then converts the model into a Tensorflow Lite flatbuffer
with dynamic range quantization. Finally, it checks the
accuracy of the converted model and compares it to the original float model.
````

`tflite__post_training_quantization#000`

````
Post-training quantization

Post-training quantization is a conversion technique that can reduce model size
while also improving CPU and hardware accelerator latency, with little
degradation in model accuracy. You can quantize an already-trained float
TensorFlow model when you convert it to TensorFlow Lite format using the
[TensorFlow Lite Converter](../models/convert/).

Note: The procedures on this page require TensorFlow 1.15 or higher.

Optimization Methods

There are several post-training quantization options to choose from. Here is a
summary table of the choices and the benefits they provide:

| Technique            | Benefits                  | Hardware         |
| -------------------- | ------------------------- | ---------------- |
| Dynamic range        | 4x smaller, 2x-3x speedup | CPU              |
: quantization         :                           :                  :
| Full integer         | 4x smaller, 3x+ speedup   | CPU, Edge TPU,   |
: quantization         :                           : Microcontrollers :
| Float16 quantization | 2x smaller, GPU           | CPU, GPU         |
:                      : acceleration              :                  :

The following decision tree can help determine which post-training quantization
method is best for your use case:

![post-training optimization options](images/optimization.jpg)
````

`tflite__post_training_quant#005`

````
Run the TFLite models

Run the TensorFlow Lite model using the Python TensorFlow Lite
Interpreter.

Load the model into an interpreter

```python
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()
```

```python
interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
interpreter_quant.allocate_tensors()
```

Test the model on one image

```python
test_image = np.expand_dims(test_images[0], axis=0).astype(np.float32)

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

interpreter.set_tensor(input_index, test_image)
interpreter.invoke()
predictions = interpreter.get_tensor(output_index)
```

```python
import matplotlib.pylab as plt

plt.imshow(test_images[0])
template = "True:{true}, predicted:{predict}"
_ = plt.title(template.format(true= str(test_labels[0]),
                              predict=str(np.argmax(predictions[0]))))
plt.grid(False)
```
````

---

## 10

**Question**

What is the difference between TFLite integer quantization with float fallback and integer-only quantization?

> TFLite 的 integer quantization with float fallback 與 integer-only quantization 有何差異？

**Answer**

The difference between TFLite integer quantization with float fallback and integer-only quantization lies in the compatibility of the resulting model with different hardware. In the case of integer with float fallback, the model uses float operators for operations that do not have integer implementations, ensuring smooth conversion to float when needed. This approach makes the model compatible with a wider range of devices and accelerators, including those that support float inputs and outputs. On the other hand, integer-only quantization enforces all operations to be quantized to integer types, specifically using tf.int8 or tf.uint8 for both input and output types. This makes the model compatible only with hardware that supports integer inputs and outputs, such as 8-bit microcontrollers and Coral Edge TPUs.

> 差異在於產出的模型與不同硬體的相容性。Integer with float fallback 對沒有 integer 實作的運算使用 float 運算子，確保需要時能順利轉成 float。這讓模型相容於更廣泛的裝置與加速器，包括支援 float 輸入與輸出的裝置。Integer-only 量化則強制所有運算量化成 integer 型別，input 與 output 都使用 tf.int8 或 tf.uint8，因此模型只相容於支援 integer 輸入與輸出的硬體，例如 8-bit microcontroller 與 Coral Edge TPU。

**Context**（2 段）

`tflite__post_training_quantization#004`

````
Integer with float fallback (using default float input/output)

In order to fully integer quantize a model, but use float operators when they
don't have an integer implementation (to ensure conversion occurs smoothly), use
the following steps:

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
<b>converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset</b>
tflite_quant_model = converter.convert()
</pre>

Note: This `tflite_quant_model` won't be compatible with integer only devices
(such as 8-bit microcontrollers) and accelerators (such as the Coral Edge TPU)
because the input and output still remain float in order to have the same
interface as the original float only model.
````

`tflite__post_training_quantization#005`

````
Integer only

*Creating integer only models is a common use case for
[TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
and [Coral Edge TPUs](https://coral.ai/).*

Note: Starting TensorFlow 2.3.0, we support the `inference_input_type` and
`inference_output_type` attributes.

Additionally, to ensure compatibility with integer only devices (such as 8-bit
microcontrollers) and accelerators (such as the Coral Edge TPU), you can enforce
full integer quantization for all ops including the input and output, by using
the following steps:

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
<b>converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]</b>
<b>converter.inference_input_type = tf.int8</b>  # or tf.uint8
<b>converter.inference_output_type = tf.int8</b>  # or tf.uint8
tflite_quant_model = converter.convert()
</pre>
````

---
