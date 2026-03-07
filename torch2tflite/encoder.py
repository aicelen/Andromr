import torch
import ai_edge_torch
from training.architecture.transformer.encoder import get_encoder
from homr.transformer.configs import Config
from torch2tflite.quantize import quant_int8
from training.onnx.split_weights import split_weights


def convert_encoder(model_version: int):
    config = Config()
    encoder = get_encoder(config)
    split_weights(f"pytorch_weights_{model_version}.pth")
    encoder.load_state_dict(torch.load("encoder_weights.pt", map_location="cpu"))
    sample_inputs = (torch.randn(1, 1, config.max_height, config.max_width),)

    # Convert and serialize PyTorch model to a tflite flatbuffer. Note that we
    # are setting the model to evaluation mode prior to conversion.
    edge_model = ai_edge_torch.convert(encoder.eval(), sample_inputs)
    edge_model.export(f"encoder_{model_version}.tflite")

    quant_int8(f"encoder_{model_version}")


if __name__ == "__main__":
    convert_encoder(331)
