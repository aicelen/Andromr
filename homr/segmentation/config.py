import os

script_location = os.path.dirname(os.path.realpath(__file__))

segnet_path_tflite = os.path.join(
    script_location, "segnet_155_wi_8_afp32.tflite"
)
segnet_path_torch = os.path.join(
    os.getcwd(),
    "training",
    "architecture",
    "segmentation",
    "segnet_155-1240eedca553155b3c75fc9c7f643465383430a0.pth",
)

segnet_version = 155

segmentation_version = segnet_version
