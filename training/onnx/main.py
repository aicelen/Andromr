# ruff: noqa: T201

import os

from homr.transformer.configs import Config
from training.onnx.convert import convert_decoder
from training.onnx.quantization import quantization_int8
from training.onnx.simplify import main as simplify_onnx_model
from training.onnx.split_weights import split_weights


def convert_transformer_quant() -> None:
    """
    Converts and Quantizes the Transformer (results in an encoder and a decoder).
    """
    config = Config()
    split_weights("pytorch_weights_331.pth")
    path_to_decoder = convert_decoder()
    simplify_onnx_model(path_to_decoder)
    print(path_to_decoder)

    path_to_decoder_int8 = quantization_int8(path_to_decoder, config.filepaths.decoder_path)
    print(path_to_decoder_int8)
    os.remove("decoder_weights.pt")
    os.remove("encoder_weights.pt")


if __name__ == "__main__":
    # Converts pytorch models used by homr to onnx
    convert_transformer_quant()
