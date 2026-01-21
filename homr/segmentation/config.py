import os
from kivy.utils import platform

from globals import APP_STORAGE

segnet_path_onnx = os.path.join(
    APP_STORAGE, "segnet_155-1240eedca553155b3c75fc9c7f643465383430a0.onnx"
)

segnet_path_tflite = os.path.join(APP_STORAGE, "segnet_155_fp16.tflite")

segnet_path_torch = os.path.join(
    os.getcwd(),
    "training",
    "architecture",
    "segmentation",
    "segnet_155-1240eedca553155b3c75fc9c7f643465383430a0.pth",
)

segnet_version = 155

segmentation_version = segnet_version
