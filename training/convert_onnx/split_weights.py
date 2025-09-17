import os
from typing import Any

import torch

from homr.transformer.configs import FilePaths


def split_weights_encoder(input_path: str) -> None:
    """
    Splits weights of the Encoder between the Transformer and the CNN-Embedder
    Args:
        input_path(str): Path to Encoder weights
    """
    # Load model
    state_dict = torch.load(input_path, map_location=torch.device("cpu"))

    cnn_encoder = {}
    transformer_encoder = {}

    for key, value in state_dict.items():
        if key.startswith("patch_embed."):
            # Remove encoder starting
            key = key.replace("patch_embed.", "")
            #key = key.replace("backbone.", "")
            # And add it to a seperate dict
            cnn_encoder[key] = value

        elif key.startswith("blocks.") or key.startswith("norm.") or key.startswith("pos_embed") or key.startswith("cls_token"):
            key = f"encoder.{key}"
            transformer_encoder[key] = value
        else:
            print(key)

    # Encoder weights can be saved directly
    torch.save(cnn_encoder, "cnn_encoder.pt")

    # Decoder weights need to be processed once more
    torch.save(transformer_encoder, "transformer_encoder.pt")


def split_weights(input_path: str) -> None:
    """
    Splits weights between Encoder and Decoder
    Args:
        input_path(str): Path to complete weights
    """
    # Load model
    state_dict = torch.load(input_path, map_location=torch.device("cpu"))

    encoder_state_dict = {}
    decoder_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("encoder."):
            # Remove encoder starting
            new_key = key.replace("encoder.", "")
            # And add it to a seperate dict
            encoder_state_dict[new_key] = value

        elif key.startswith("decoder."):
            # Remove decoder starting
            new_key = key.replace("decoder.", "")
            # And add it to a seperate dict
            decoder_state_dict[new_key] = value

    # Encoder weights can be saved directly
    torch.save(encoder_state_dict, "encoder_weights.pt")

    # Decoder weights need to be processed once more
    torch.save(remove_score_decoder_weights(decoder_state_dict), "decoder_weights.pt")


def remove_score_decoder_weights(full_state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Since get_decoder_onnx() is not using ScoreDecoder() we need to change the weights
    so they are directly put into ScoreTransformerWrapper().

    Args:
        full_state_dict (dict): Model dictionary
    """
    # Remove the net. starting
    transformer_state_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith("net."):
            new_key = key.replace("net.", "")
            transformer_state_dict[new_key] = value

    return transformer_state_dict
