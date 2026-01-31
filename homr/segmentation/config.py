import os

from globals import MODEL_STORAGE

segnet_path_tflite = os.path.join(MODEL_STORAGE, "segnet_308_int8.tflite")

segnet_path_torch = os.path.join(
    os.getcwd(),
    "training",
    "architecture",
    "segmentation",
    "segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.pth",
)

segnet_version = 308

segmentation_version = segnet_version
