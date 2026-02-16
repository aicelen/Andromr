import os

from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process


def quantization_int8(
    model_path: str, out_path: str | None = None
) -> None:
    """
    Dynamic Quantization of an onnx model to int8
    Args:
        model_path(str): Path to onnx model
        out_path(str): Path for saving the quantized model
    """
    if out_path is None:
        out_path = model_path

    quant_pre_process(
        model_path, "model_preprocessed.onnx"
    )  # Preprocess model for better quantization results
    quantize_dynamic("model_preprocessed.onnx", out_path, weight_type=QuantType.QInt8)

    os.remove("model_preprocessed.onnx")
    return out_path
