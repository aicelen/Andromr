from ai_edge_quantizer import quantizer
from ai_edge_quantizer import recipe


def quant_int8(model_path):
    qt = quantizer.Quantizer(f"{model_path}.tflite")
    qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
    qt.quantize().export_model(f"{model_path}_int8.tflite")
