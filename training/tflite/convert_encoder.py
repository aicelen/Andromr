import torch
import ai_edge_torch  # Use ai_edge_torch_nightly
from training.architecture.transformer.encoder import get_encoder
from homr.transformer.configs import Config
from tflite.quantize import quant_int8


def convert_encoder(model_name):
    config = Config()
    encoder = get_encoder(config)
    encoder.load_state_dict(torch.load("encoder_weights.pt", map_location="cpu"))
    sample_inputs = (torch.randn(1, 1, config.max_height, config.max_width),)

    # Convert and serialize PyTorch model to a tflite flatbuffer. Note that we
    # are setting the model to evaluation mode prior to conversion.
    edge_model = ai_edge_torch.convert(encoder.eval(), sample_inputs)
    edge_model.export(f"{model_name}.tflite")

    quant_int8(model_name)


if __name__ == "__main__":
    convert_encoder("encoder_242")
