import os

import torch

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from training.architecture.transformer.decoder import (
    ScoreTransformerWrapper,
    get_score_wrapper,
    init_cache,
)
from training.architecture.transformer.encoder import get_encoder


class DecoderWrapper(torch.nn.Module):
    def __init__(self, model: ScoreTransformerWrapper) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        articulations: torch.Tensor,
        context: torch.Tensor,
        cache_len: torch.Tensor,
        *cache: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
    ]:
        (
            out_rhythms,
            out_pitchs,
            out_lifts,
            out_positions,
            out_articulations,
            _x,
            attention,
            *cache,
        ) = self.model(
            rhythms=rhythms,
            pitchs=pitchs,
            lifts=lifts,
            articulations=articulations,
            context=context,
            cache_len=cache_len,
            mask=None,
            cache=cache,
            return_center_of_attention=False,
        )
        return (
            out_rhythms,
            out_pitchs,
            out_lifts,
            out_positions,
            out_articulations,
            attention,
            *cache,
        )


def convert_encoder() -> str:
    raise NotImplementedError("Please convert the encoder to tflite not onnx.")

def convert_decoder() -> str:
    """
    Converts the decoder to onnx.
    """
    config = Config()
    model = get_score_wrapper(config, attn_flash=True)
    model.eval()

    path_out = config.filepaths.decoder_path

    if os.path.exists(path_out):
        eprint(path_out, "is already present")
        return path_out

    model.load_state_dict(
        torch.load(r"decoder_weights.pt", weights_only=True, map_location=torch.device("cpu")),
        strict=True,
    )

    # Using a wrapper model with a custom forward() function
    wrapped_model = DecoderWrapper(model)
    wrapped_model.eval()

    # Create input data
    # Mask is not used since it caused problems with the tensor size
    kv_cache, kv_input_names, kv_output_names, dynamic_axes, cache_length = init_cache()
    rhythms = torch.randint(0, config.num_rhythm_tokens, (1, 1)).long()
    pitchs = torch.randint(0, config.num_pitch_tokens, (1, 1)).long()
    lifts = torch.randint(0, config.num_lift_tokens, (1, 1)).long()
    articulations = torch.randint(0, config.num_articulation_tokens, (1, 1)).long()
    cache_len = torch.tensor([cache_length]).long()
    cache = kv_cache
    context = torch.randn((1, 1281, 312)).float()

    dynamic_axes["context"] = {1: "cache_exists"}

    torch.onnx.export(
        wrapped_model,
        (rhythms, pitchs, lifts, articulations, context, cache_len, *cache),
        path_out,
        input_names=[
            "rhythms",
            "pitchs",
            "lifts",
            "articulations",
            "context",
            "cache_len",
            *kv_input_names,
        ],
        output_names=[
            "out_rhythms",
            "out_pitchs",
            "out_lifts",
            "out_positions",
            "out_articulations",
            *kv_output_names,
        ],
        dynamic_axes=dynamic_axes,
        opset_version=18,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )
    return path_out


def convert_segnet() -> str:
    """
    Converts the segnet model to onnx.
    """
    raise NotImplementedError("Please convert the segnet to tflite not onnx.")