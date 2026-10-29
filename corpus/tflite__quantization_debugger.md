<!-- source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/quantization_debugger.ipynb -->

##### Copyright 2024 The AI Edge Authors.

# Inspecting Quantization Errors with Quantization Debugger

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/lite/performance/quantization_debugger"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/quantization_debugger.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/quantization_debugger.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/tensorflow/tensorflow/lite/g3doc/performance/quantization_debugger.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
  <td>
    <a href="https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5"><img src="https://www.tensorflow.org/images/hub_logo_32px.png" />See TF Hub model</a>
  </td>
</table>

Although full-integer quantization provides improved model size and latency, the
quantized model won't always work as expected. It's usually expected for the
model quality (e.g. accuracy, mAP, WER) to be slightly lower than the original
float model. However, there are cases where the model quality can go below your
expectations or generates completely wrong results.

When this problem happens, it's tricky and painful to spot the root cause of the
quantization error, and it's even more difficult to fix the quantization error.
To assist this model inspection process, a **quantization debugger** can be used
to identify problematic layers, and **selective quantization** can leave those
problematic layers in float so that the model accuracy can be recovered at the
cost of reduced benefit from quantization.

Note: This API is experimental, and there might be breaking changes in the API
in the course of improvements.

## Quantization Debugger

Quantization debugger makes it possible to do quantization quality metric
analysis in the existing model. Quantization debugger can automate processes for
running model with a debug dataset, and collecting quantization quality metrics
for each tensors.

Note: Quantization debugger and selective quantization currently only works for
full-integer quantization with int8 activations.

### Prerequisites

If you already have a pipeline to quantize a model, you have all necessary
pieces to run quantization debugger!

*   Model to quantize
*   Representative dataset

In addition to the model and data, you will need to use a data processing framework
(e.g. pandas, Google Sheets) to analyze the exported results.

### Setup

This section prepares libraries, MobileNet v3 model, and a test dataset of 100
images.

```python
# Quantization debugger is available from TensorFlow 2.7.0
!pip uninstall -y tensorflow
!pip install tf-nightly
!pip install tensorflow_datasets --upgrade  # imagenet_v2 needs latest checksum
```

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
```

```python
#@title Boilerplates and helpers
MODEL_URI = 'https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5'


def process_image(data):
  data['image'] = tf.image.resize(data['image'], (224, 224)) / 255.0
  return data


# Representative dataset
def representative_dataset(dataset):

  def _data_gen():
    for data in dataset.batch(1):
      yield [data['image']]

  return _data_gen


def eval_tflite(tflite_model, dataset):
  """Evaluates tensorflow lite classification model with the given dataset."""
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()

  input_idx = interpreter.get_input_details()[0]['index']
  output_idx = interpreter.get_output_details()[0]['index']

  results = []

  for data in representative_dataset(dataset)():
    interpreter.set_tensor(input_idx, data[0])
    interpreter.invoke()
    results.append(interpreter.get_tensor(output_idx).flatten())

  results = np.array(results)
  gt_labels = np.array(list(dataset.map(lambda data: data['label'] + 1)))
  accuracy = (
      np.sum(np.argsort(results, axis=1)[:, -5:] == gt_labels.reshape(-1, 1)) /
      gt_labels.size)
  print(f'Top-5 accuracy (quantized): {accuracy * 100:.2f}%')


model = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(224, 224, 3), batch_size=1),
  hub.KerasLayer(MODEL_URI)
])
model.compile(
    loss='sparse_categorical_crossentropy',
    metrics='sparse_top_k_categorical_accuracy')
model.build([1, 224, 224, 3])

# Prepare dataset with 100 examples
ds = tfds.load('imagenet_v2', split='test[:1%]')
ds = ds.map(process_image)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.representative_dataset = representative_dataset(ds)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()
```

```python
test_ds = ds.map(lambda data: (data['image'], data['label'] + 1)).batch(16)
loss, acc = model.evaluate(test_ds)
print(f'Top-5 accuracy (float): {acc * 100:.2f}%')
```

```python
eval_tflite(quantized_model, ds)
```

We can see that the original model has a much higher top-5 accuracy for our
small dataset, while the quantized model has a significant accuracy loss.

### Step 1. Debugger preparation

The Easiest way to use the quantization debugger is to provide
`tf.lite.TFLiteConverter` that you have been using to quantize the model.

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset(ds)

# my_debug_dataset should have the same format as my_representative_dataset
debugger = tf.lite.experimental.QuantizationDebugger(
    converter=converter, debug_dataset=representative_dataset(ds))
```

### Step 2. Running the debugger and getting the results

When you call `QuantizationDebugger.run()`, the debugger will log differences
between float tensors and quantized tensors for the same op location, and
process them with given metrics.

```python
debugger.run()
```

The processed metrics can be accessed with
`QuantizationDebugger.layer_statistics`, or can be dumped to a text file in CSV
format with `QuantizationDebugger.layer_statistics_dump()`.

```python
RESULTS_FILE = '/tmp/debugger_results.csv'
with open(RESULTS_FILE, 'w') as f:
  debugger.layer_statistics_dump(f)
```

```python
!head /tmp/debugger_results.csv
```

For each row in the dump, the op name and index comes first, followed by
quantization parameters and error metrics (including
[user-defined error metrics](#custom-metrics), if any). The resulting CSV file
can be used to pick problematic layers with large quantization error metrics.

With pandas or other data processing libraries, we can inspect detailed
per-layer error metrics.

```python
layer_stats = pd.read_csv(RESULTS_FILE)
layer_stats.head()
```

### Step 3. Data analysis

There are various ways to analyze the resulting. First, let's add some useful
metrics derived from the debugger's outputs. (`scale` means the quantization
scale factor for each tensor.)

*   Range (`256 / scale`)
*   RMSE / scale (`sqrt(mean_squared_error) / scale`)

The `RMSE / scale` is close to `1 / sqrt(12)` (~ 0.289) when quantized
distribution is similar to the original float distribution, indicating a good
quantized model. The larger the value is, it's more likely for the layer not
being quantized well.

```python
layer_stats['range'] = 255.0 * layer_stats['scale']
layer_stats['rmse/scale'] = layer_stats.apply(
    lambda row: np.sqrt(row['mean_squared_error']) / row['scale'], axis=1)
layer_stats[['op_name', 'range', 'rmse/scale']].head()
```

```python
plt.figure(figsize=(15, 5))
ax1 = plt.subplot(121)
ax1.bar(np.arange(len(layer_stats)), layer_stats['range'])
ax1.set_ylabel('range')
ax2 = plt.subplot(122)
ax2.bar(np.arange(len(layer_stats)), layer_stats['rmse/scale'])
ax2.set_ylabel('rmse/scale')
plt.show()
```

There are many layers with wide ranges, and some layers that have high
`RMSE/scale` values. Let's get the layers with high error metrics.

```python
layer_stats[layer_stats['rmse/scale'] > 0.7][[
    'op_name', 'range', 'rmse/scale', 'tensor_name'
]]
```

With these layers, you can try selective quantization to see if not quantizing
those layers improves model quality.

```python
suspected_layers = list(
    layer_stats[layer_stats['rmse/scale'] > 0.7]['tensor_name'])
```

In addition to these, skipping quantization for the first few layers also helps
improving quantized model's quality.

```python
suspected_layers.extend(list(layer_stats[:5]['tensor_name']))
```

## Selective Quantization

Selective quantization skips quantization for some nodes, so that the
calculation can happen in the original floating-point domain. When correct
layers are skipped, we can expect some model quality recovery at the cost of
increased latency and model size.

However, if you're planning to run quantized models on integer-only accelerators
(e.g. Hexagon DSP, EdgeTPU), selective quantization would cause fragmentation of
the model and would result in slower inference latency mainly caused by data
transfer cost between CPU and those accelerators. To prevent this, you can
consider running
[quantization aware training](https://www.tensorflow.org/model_optimization/guide/quantization/training)
to keep all the layers in integer while preserving the model accuracy.

Quantization debugger's option accepts `denylisted_nodes` and `denylisted_ops`
options for skipping quantization for specific layers, or all instances of
specific ops. Using the `suspected_layers` we prepared from the previous step, we
can use a quantization debugger to get a selectively quantized model.

```python
debug_options = tf.lite.experimental.QuantizationDebugOptions(
    denylisted_nodes=suspected_layers)
debugger = tf.lite.experimental.QuantizationDebugger(
    converter=converter,
    debug_dataset=representative_dataset(ds),
    debug_options=debug_options)
```

```python
selective_quantized_model = debugger.get_nondebug_quantized_model()
eval_tflite(selective_quantized_model, ds)
```

The accuracy is still lower compared to the original float model, but we have
notable improvement from the whole quantized model by skipping quantization for
~10 layers out of 111 layers.

You can also try to not quantize all ops in the same class. For example, to
skip quantization for all mean ops, you can pass `MEAN` to `denylisted_ops`.

```python
debug_options = tf.lite.experimental.QuantizationDebugOptions(
    denylisted_ops=['MEAN'])
debugger = tf.lite.experimental.QuantizationDebugger(
    converter=converter,
    debug_dataset=representative_dataset(ds),
    debug_options=debug_options)
```

```python
selective_quantized_model = debugger.get_nondebug_quantized_model()
eval_tflite(selective_quantized_model, ds)
```

With these techniques, we are able to improve the quantized MobileNet V3 model
accuracy. Next we'll explore advanced techniques to improve the model accuracy
even more.

## Advanced usages

With the following features, you can further customize your debugging pipeline.

### Custom metrics

By default, the quantization debugger emits five metrics for each float-quant
difference: tensor size, standard deviation, mean error, max absolute error, and
mean squared error. You can add more custom metrics by passing them to options.
For each metrics, the result should be a single float value and the resulting
metric will be an average of metrics from all examples.

*   `layer_debug_metrics`: calculate metric based on diff for each op outputs
    from float and quantized op outputs.
*   `layer_direct_compare_metrics`: rather than getting diff only, this will
    calculate metric based on raw float and quantized tensors, and its
    quantization parameters (scale, zero point)
*   `model_debug_metrics`: **only used when `float_model_(path|content)` is
    passed** to the debugger. In addition to the op-level metrics, the final layer
    output is compared to the reference output from the original float model.

```python
debug_options = tf.lite.experimental.QuantizationDebugOptions(
    layer_debug_metrics={
        'mean_abs_error': (lambda diff: np.mean(np.abs(diff)))
    },
    layer_direct_compare_metrics={
        'correlation':
            lambda f, q, s, zp: (np.corrcoef(f.flatten(),
                                             (q.flatten() - zp) / s)[0, 1])
    },
    model_debug_metrics={
        'argmax_accuracy': (lambda f, q: np.mean(np.argmax(f) == np.argmax(q)))
    })

debugger = tf.lite.experimental.QuantizationDebugger(
    converter=converter,
    debug_dataset=representative_dataset(ds),
    debug_options=debug_options)
```

```python
debugger.run()
```

```python
CUSTOM_RESULTS_FILE = '/tmp/debugger_results.csv'
with open(CUSTOM_RESULTS_FILE, 'w') as f:
  debugger.layer_statistics_dump(f)

custom_layer_stats = pd.read_csv(CUSTOM_RESULTS_FILE)
custom_layer_stats[['op_name', 'mean_abs_error', 'correlation']].tail()
```

The result of `model_debug_metrics` can be separately seen from
`debugger.model_statistics`.

```python
debugger.model_statistics
```

### Using (internal) mlir_quantize API to access in-depth features

Note: Some features in the following section,
`TFLiteConverter._experimental_calibrate_only` and `converter.mlir_quantize` are
experimental internal APIs, and subject to change in a non-backward compatible
way.

```python
from tensorflow.lite.python import convert
```

#### Whole model verify mode

The default behavior for the debug model generation is per-layer verify. In this
mode, the input for float and quantize op pair is from the same source (previous
quantized op). Another mode is whole-model verify, where the float and quantize
models are separated. This mode would be useful to observe how the error is
being propagated down the model. To enable, `enable_whole_model_verify=True` to
`convert.mlir_quantize` while generating the debug model manually.

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.representative_dataset = representative_dataset(ds)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter._experimental_calibrate_only = True
calibrated_model = converter.convert()
```

```python
# Note that enable_numeric_verify and enable_whole_model_verify are set.
quantized_model = convert.mlir_quantize(
    calibrated_model,
    enable_numeric_verify=True,
    enable_whole_model_verify=True)
debugger = tf.lite.experimental.QuantizationDebugger(
    quant_debug_model_content=quantized_model,
    debug_dataset=representative_dataset(ds))
```

#### Selective quantization from an already calibrated model

You can directly call `convert.mlir_quantize` to get the selective quantized
model from already calibrated model. This would be particularly useful when you
want to calibrate the model once, and experiment with various denylist
combinations.

```python
selective_quantized_model = convert.mlir_quantize(
    calibrated_model, denylisted_nodes=suspected_layers)
eval_tflite(selective_quantized_model, ds)
```
