from pathlib import Path
from time import perf_counter
from math import ceil

import cv2
import numpy as np

from homr.inference_engine import TensorFlowModel
from homr.segmentation.config import segnet_path_tflite
from homr.simple_logging import eprint
from homr.type_definitions import NDArray
from globals import appdata

model = None


def load_segnet(num_threads: int = appdata.threads, use_gpu: bool = appdata.gpu):
    global model
    if model is None:
        model = TensorFlowModel(segnet_path_tflite, num_threads=num_threads, use_gpu=use_gpu)


class ExtractResult:
    def __init__(
        self,
        filename: Path,
        original: NDArray,
        staff: NDArray,
        symbols: NDArray,
        stems_rests: NDArray,
        notehead: NDArray,
        clefs_keys: NDArray,
    ):
        self.filename = filename
        self.original = original
        self.staff = staff
        self.symbols = symbols
        self.stems_rests = stems_rests
        self.notehead = notehead
        self.clefs_keys = clefs_keys


def merge_patches(
    patches: list[NDArray],
    image_shape: tuple[int, int],
    win_size: int,
    step_size: int = -1,
) -> NDArray:
    reconstructed = np.zeros(image_shape, dtype=np.float32)
    weight = np.zeros(image_shape, dtype=np.float32)

    idx = 0
    for iy in range(0, image_shape[0], step_size):
        if iy + win_size > image_shape[0]:
            y = image_shape[0] - win_size
        else:
            y = iy
        for ix in range(0, image_shape[1], step_size):
            if ix + win_size > image_shape[1]:
                x = image_shape[1] - win_size
            else:
                x = ix

            reconstructed[y : y + win_size, x : x + win_size] += patches[idx]
            weight[y : y + win_size, x : x + win_size] += 1
            idx += 1

    # Avoid division by zero
    weight[weight == 0] = 1
    reconstructed /= weight

    return reconstructed.astype(patches[0].dtype)


def inference(
    image_org: NDArray, step_size: int, win_size: int
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Inference function for the segementation model.
    Args:
        image_org(NDArray): Array of the input image
        step_size(int): How far the window moves between to input images.
        win_size(int): Debug only.

    Returns:
        ExtractResult class.
    """
    eprint("Starting Inference.")
    global model
    load_segnet()

    t0 = perf_counter()
    num_steps = ceil(image_org.shape[0] / step_size) * ceil(image_org.shape[1] / step_size)
    progress_increment = 100 / num_steps

    data = []
    image = image_org.astype(np.float32)
    for y_loop in range(0, image.shape[0], step_size):
        if y_loop + win_size > image.shape[0]:
            y = image.shape[0] - win_size
        else:
            y = y_loop
        for x_loop in range(0, image.shape[1], step_size):
            if x_loop + win_size > image.shape[1]:
                x = image.shape[1] - win_size
            else:
                x = x_loop
            hop = image[y : y + win_size, x : x + win_size, :]

            hop = np.expand_dims(hop, axis=0)
            out = model.run(hop)
            out_filtered = np.argmax(out, axis=-1)
            out_filtered = np.squeeze(out_filtered, axis=0)
            data.append(out_filtered)

            # Update progress bar value
            appdata.homr_progress += progress_increment

    eprint(f"Segnet Inference time: {perf_counter() - t0}")

    merged = merge_patches(
        data, (int(image_org.shape[0]), int(image_org.shape[1])), win_size, step_size
    )
    stems_layer = 1
    stems_rests = np.where(merged == stems_layer, 1, 0)
    notehead_layer = 2
    notehead = np.where(merged == notehead_layer, 1, 0)
    clefs_keys_layer = 3
    clefs_keys = np.where(merged == clefs_keys_layer, 1, 0)
    staff_layer = 4
    staff = np.where(merged == staff_layer, 1, 0)
    symbol_layer = 5
    symbols = np.where(merged == symbol_layer, 1, 0)

    return staff, symbols, stems_rests, notehead, clefs_keys


def extract(
    original_image: NDArray,
    img_path_str: str,
    use_cache: bool = False,
    step_size: int = -1,
    win_size: int = 320,
) -> ExtractResult:
    img_path = Path(img_path_str)
    staff, symbols, stems_rests, notehead, clefs_keys = inference(
        original_image,
        step_size=step_size,
        win_size=win_size,
    )
    original_image = cv2.resize(original_image, (staff.shape[1], staff.shape[0]))

    return ExtractResult(
        img_path, original_image, staff, symbols, stems_rests, notehead, clefs_keys
    )
