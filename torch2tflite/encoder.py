import torch
import ai_edge_torch
from training.architecture.transformer.encoder import get_encoder
from homr.transformer.configs import Config
from torch2tflite.quantize import quant_int8
from training.onnx.split_weights import split_weights
import argparse
from homr.transformer.configs import default_config

def convert_encoder(use_split_weights: bool):
    config = Config()
    encoder = get_encoder(config)
    model_version = default_config.filepaths.model_name
    if use_split_weights:
        split_weights(f"pytorch_weights_{model_version}.pth")
        encoder.load_state_dict(torch.load("encoder_weights.pt", map_location="cpu"))
    else:
        encoder.load_state_dict(torch.load(f"encoder_weights_{model_version}.pt", map_location="cpu"))

    sample_inputs = (torch.randn(1, 1, config.max_height, config.max_width),)

    # Convert and serialize PyTorch model to a tflite flatbuffer. Note that we
    # are setting the model to evaluation mode prior to conversion.
    edge_model = ai_edge_torch.convert(encoder.eval(), sample_inputs)
    edge_model.export(f"encoder_{model_version}.tflite")

    quant_int8(f"encoder_{model_version}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_weights", action="store_true", help="Overwrite existing models")

    args = parser.parse_args()

    convert_encoder(args.split_weights)


if __name__ == "__main__":
    main()
