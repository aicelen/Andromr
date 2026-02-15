# ruff: noqa: T201

import os

from training.onnx.convert import (
    convert_decoder,
    convert_segnet
)
from training.onnx.quantization import quantization_int8
from training.onnx.simplify import main as simplify_onnx_model
from training.onnx.split_weights import split_weights
from homr.transformer.configs import Config


def onnx_decoder(transformer_path: str | None = None) -> None:
    if transformer_path is None:
        raise FileExistsError("You did not specify the path of your pytorch models")

    split_weights(transformer_path)

    path_to_decoder = convert_decoder()
    simplify_onnx_model(path_to_decoder)
    quantization_int8(path_to_decoder)
    simplify_onnx_model(path_to_decoder)

    os.remove("decoder_weights.pt")
    os.remove("encoder_weights.pt")


if __name__ == "__main__":
    # Converts pytorch models used by homr to onnx
    onnx_decoder(Config().filepaths.checkpoint)
