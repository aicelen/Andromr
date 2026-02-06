from homr.inference_engine import OnnxModel, TensorFlowModel
import os
import numpy as np

onnx = OnnxModel("encoder_pytorch_model_242-a00be6debbedf617acdf39558c89ba6113c06af3.onnx")
onnx.load()

tflite = TensorFlowModel(os.path.join("data", "models", "encoder_pytorch_model_242-a00be6debbedf617acdf39558c89ba6113c06af3.tflite"))
tflite.load()


input = np.random.rand(1, 1, 256, 1280).astype(np.float32)


onnx_out = onnx.run({"input": input}, {"output": "output"})
tflite_out = tflite.run(input)

print(np.median((onnx_out - tflite_out)))
print(np.average((onnx_out - tflite_out)))

data_range = np.max(onnx_out) - np.min(onnx_out)

# Calculate error relative to the range
relative_error = np.average((onnx_out - tflite_out)) / data_range
print(f"Error is {relative_error:.4%} of the total data range.")