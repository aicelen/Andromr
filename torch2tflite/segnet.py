import torch
import ai_edge_torch
from training.architecture.segmentation.model import create_segnet
from homr.transformer.configs import Config
from torch2tflite.quantize import quant_int8

def convert_segnet(model_name):
    # Use resnet18 with pre-trained weights.
    segnet = create_segnet()
    segnet.load_state_dict(torch.load("segnet_308.pth", map_location="cpu"))

    sample_inputs = (torch.randn(1, 3, 320, 320),)

    # Convert and serialize PyTorch model to a tflite flatbuffer. Note that we
    # are setting the model to evaluation mode prior to conversion.
    edge_model = ai_edge_torch.convert(segnet.eval(), sample_inputs)
    edge_model.export(f"{model_name}.tflite")

    quant_int8(model_name)

convert_segnet("segnet_308")