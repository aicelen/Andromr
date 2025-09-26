import os

import torch

from homr.segmentation.config import segnet_path_torch
from homr.transformer.configs import Config
from training.architecture.segmentation.model import create_segnet  # type: ignore
from training.architecture.transformer.decoder import (
    ScoreTransformerWrapper,
    get_decoder_onnx,
)
from training.architecture.transformer.encoder import (
    get_encoder,
    get_backbone,
    get_transformer,
)
from training.convert_onnx.simplify import main as simplify_onnx_model


class DecoderWrapper(torch.nn.Module):
    def __init__(self, model: ScoreTransformerWrapper) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        result = self.model(rhythms, pitchs, lifts, mask=None, context=context)
        return result


def convert_encoder() -> str:
    """
    Converts the encoder to onnx
    """
    config = Config()

    dir_path = os.path.dirname(config.filepaths.checkpoint)
    filename = os.path.splitext(os.path.basename(config.filepaths.checkpoint))[0]
    path_out = os.path.join(dir_path, f"encoder_{filename}.onnx")

    # Get Encoder
    model = get_encoder(config)

    # Load weights
    model.load_state_dict(
        torch.load(
            r"encoder_weights.pt", weights_only=True, map_location=torch.device("cpu")
        ),
        strict=True,
    )

    # Set eval mode
    model.eval()

    # Prepare input tensor
    input_tensor = torch.randn(1, 1, 128, 1280).float()

    # Export to onnx
    torch.onnx.export(
        model,
        input_tensor,  # type: ignore
        path_out,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    return path_out


def convert_decoder() -> str:
    """
    Converts the decoder to onnx.
    """
    config = Config()
    model = get_decoder_onnx(config)
    model.eval()

    dir_path = os.path.dirname(config.filepaths.checkpoint)
    filename = os.path.splitext(os.path.basename(config.filepaths.checkpoint))[0]
    path_out = os.path.join(dir_path, f"decoder_{filename}.onnx")

    model.load_state_dict(
        torch.load(
            r"decoder_weights.pt", weights_only=True, map_location=torch.device("cpu")
        ),
        strict=True,
    )

    # Using a wrapper model with a custom forward() function
    wrapped_model = DecoderWrapper(model)
    wrapped_model.eval()

    # Create input data
    # Mask is not used since it caused problems with the tensor size
    rhythms = torch.randint(0, config.num_rhythm_tokens, (1, 10)).long()
    pitchs = torch.randint(0, config.num_pitch_tokens, (1, 10)).long()
    lifts = torch.randint(0, config.num_lift_tokens, (1, 10)).long()
    context = torch.randn((1, 641, 312)).float()

    dynamic_axes = {
        "rhythms": {0: "batch_size", 1: "input_seq_len"},
        "pitchs": {0: "batch_size", 1: "input_seq_len"},
        "lifts": {0: "batch_size", 1: "input_seq_len"},
        "context": {0: "batch_size"},
        "out_rhythms": {0: "batch_size", 1: "output_seq_len"},
        "out_pitchs": {0: "batch_size", 1: "output_seq_len"},
        "out_lifts": {0: "batch_size", 1: "output_seq_len"},
    }

    torch.onnx.export(
        wrapped_model,
        (rhythms, pitchs, lifts, context),
        path_out,
        input_names=["rhythms", "pitchs", "lifts", "context"],
        output_names=[
            "out_rhythms",
            "out_pitchs",
            "out_lifts",
        ],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
    )
    return path_out


def convert_segnet() -> str:
    """
    Converts the segnet model to onnx.
    """
    model = create_segnet()
    model.load_state_dict(torch.load(segnet_path_torch, weights_only=True), strict=True)
    model.eval()

    # Input dimension is 1x3x320x320
    sample_inputs = torch.randn(1, 3, 320, 320)

    torch.onnx.export(
        model,
        sample_inputs,  # type: ignore
        f"{os.path.splitext(segnet_path_torch)[0]}.onnx",
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        # dyamic axes are required for dynamic batch_size
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    return f"{os.path.splitext(segnet_path_torch)[0]}.onnx"


def convert_cnn_encoder():
    """
    Converts the encoder to onnx
    """
    config = Config()

    dir_path = os.path.dirname(config.filepaths.checkpoint)
    filename = os.path.splitext(os.path.basename(config.filepaths.checkpoint))[0]
    path_out = os.path.join(dir_path, f"cnn_encoder_{filename}.onnx")

    # Get Encoder
    model = get_backbone(config)

    # Load weights
    model.load_state_dict(
        torch.load(
            r"cnn_encoder.pt", weights_only=True, map_location=torch.device("cpu")
        ),
        strict=True,
    )

    # Set eval mode
    model.eval()

    # Prepare input tensor
    input_tensor = torch.randn(1, 1, 128, 1280).float()

    # Export to onnx
    torch.onnx.export(
        model,
        input_tensor,  # type: ignore
        path_out,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    return path_out


def convert_transformer_encoder():
    """
    Converts the encoder to onnx
    """
    config = Config()

    dir_path = os.path.dirname(config.filepaths.checkpoint)
    filename = os.path.splitext(os.path.basename(config.filepaths.checkpoint))[0]
    path_out = os.path.join(dir_path, f"transformer_encoder_{filename}.onnx")

    # Get Encoder
    model = get_transformer(config)

    # Load weights
    model.load_state_dict(
        torch.load(
            r"transformer_encoder.pt",
            weights_only=True,
            map_location=torch.device("cpu"),
        ),
        strict=True,
    )

    # Set eval mode
    model.eval()

    # Prepare input tensor
    input_tensor = torch.randn(1, 312, 8, 80).float()

    # Export to onnx
    torch.onnx.export(
        model,
        input_tensor,  # type: ignore
        path_out,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    return path_out


if __name__ == "__main__":
    simplify_onnx_model(convert_cnn_encoder())  # converted to tflite 0.3s
    simplify_onnx_model(
        convert_transformer_encoder()
    )  # lets use quint8 for better performance


# with all optimizations this means inference times of:
# segnet: 40 * 0.2s = 8s
# encoder cnn: 10 * 0.3s = 3s
# encoder transformer: 10 * 0.1s = 1s
# decoder: 10 * 40 * 0.05s = 20s
# so 32s which means homr will run in under 1 minute
