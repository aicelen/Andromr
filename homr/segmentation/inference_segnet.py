import hashlib
import os
from pathlib import Path
from time import perf_counter
from math import ceil

import cv2
import numpy as np
from globals import appdata

from homr.segmentation.config import segmentation_version, segnet_path_tflite
from homr.simple_logging import eprint
from homr.type_definitions import NDArray
from homr.inference_engine.tflite_model import TensorFlowModel

segnet: TensorFlowModel | None = None


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


def extract_patch(image: NDArray, y: int, x: int, win_size: int) -> NDArray:
    """
    Returns a full-size (3, win_size, win_size) patch.
    Pads with white pixes if the patch exceeds image boundaries.
    """
    c, h, w = image.shape
    patch = np.full((c, win_size, win_size), 255, dtype=image.dtype)

    y0 = max(y, 0)
    x0 = max(x, 0)
    y1 = min(y + win_size, h)
    x1 = min(x + win_size, w)

    py0 = 0
    px0 = 0
    py1 = py0 + (y1 - y0)
    px1 = px0 + (x1 - x0)

    patch[:, py0:py1, px0:px1] = image[:, y0:y1, x0:x1]
    return patch


def merge_patches(
    patches: list[NDArray], image_shape: tuple[int, int], win_size: int, step_size: int
) -> NDArray:
    reconstructed = np.zeros(image_shape, dtype=np.float32)
    weight = np.zeros(image_shape, dtype=np.float32)

    idx = 0
    for iy in range(0, image_shape[0], step_size):
        y = min(iy, image_shape[0] - win_size)
        y0 = max(y, 0)
        y1 = min(y + win_size, image_shape[0])

        for ix in range(0, image_shape[1], step_size):
            x = min(ix, image_shape[1] - win_size)
            x0 = max(x, 0)
            x1 = min(x + win_size, image_shape[1])

            patch = patches[idx]
            ph = y1 - y0
            pw = x1 - x0

            reconstructed[y0:y1, x0:x1] += patch[:ph, :pw]
            weight[y0:y1, x0:x1] += 1
            idx += 1

    # Avoid division by zero
    weight[weight == 0] = 1
    reconstructed /= weight

    return reconstructed.astype(patches[0].dtype)


def inference(
    image_org: NDArray, use_gpu_inference: bool, batch_size: int, step_size: int, win_size: int
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Inference function for the segementation model.
    Args:
        image_org(NDArray): Array of the input image
        batch_size(int): Mainly for speeding up GPU performance. Minimal impact on CPU speed.
        step_size(int): How far the window moves between to input images.
        win_size(int): Debug only.

    Returns:
        ExtractResult class.
    """
    eprint("Starting Inference.")
    t0 = perf_counter()
    if step_size < 0:
        step_size = win_size // 2

    num_steps = ceil(image_org.shape[0] / step_size) * ceil(image_org.shape[1] / step_size)
    progress_increment = 100 / num_steps

    global segnet
    if segnet is None:
        segnet = TensorFlowModel(segnet_path_tflite)

    image_org = cv2.cvtColor(image_org, cv2.COLOR_GRAY2BGR)
    image = np.transpose(image_org, (2, 0, 1)).astype(np.float32)

    c, h, w = image.shape
    data: list[NDArray] = []
    batch: list[NDArray] = []

    for y_loop in range(0, max(h, win_size), step_size):
        y = min(y_loop, h - win_size)
        for x_loop in range(0, max(w, win_size), step_size):
            x = min(x_loop, w - win_size)

            hop = extract_patch(image, y, x, win_size)

            batch.append(hop)

            if len(batch) == batch_size:
                hop = np.expand_dims(hop, axis=0)
                batch_out = segnet.run(hop, (1, 6, 320, 320))
                for out in batch_out:
                    data.append(np.argmax(out, axis=0))
                batch.clear()
            appdata.homr_progress += progress_increment

    if batch:
        batch_out = segnet.run(np.stack(batch, axis=0))
        for out in batch_out:
            data.append(np.argmax(out, axis=0))

    eprint(f"Segnet Inference time: {perf_counter() - t0}; batch_size {batch_size}")

    merged = merge_patches(
        data, (int(image_org.shape[0]), int(image_org.shape[1])), win_size, step_size
    )

    stems_rests = (merged == 1).astype(np.uint8)
    notehead = (merged == 2).astype(np.uint8)
    clefs_keys = (merged == 3).astype(np.uint8)
    staff = (merged == 4).astype(np.uint8)
    symbols = (merged == 5).astype(np.uint8)

    return staff, symbols, stems_rests, notehead, clefs_keys


def extract(
    original_image: NDArray,
    img_path_str: str,
    use_cache: bool = False,
    use_gpu_inference: bool = True,
    batch_size: int = 8,
    step_size: int = -1,
    win_size: int = 320,
) -> ExtractResult:
    img_path = Path(img_path_str)
    f_name = os.path.splitext(img_path.name)[0]
    npy_path = img_path.parent / f"{f_name}.npy"
    loaded_from_cache = False
    if not loaded_from_cache:
        staff, symbols, stems_rests, notehead, clefs_keys = inference(
            original_image,
            use_gpu_inference=use_gpu_inference,
            batch_size=1,  # Fixed batch size
            step_size=step_size,
            win_size=win_size,
        )

    original_image = cv2.resize(original_image, (staff.shape[1], staff.shape[0]))

    return ExtractResult(
        img_path, original_image, staff, symbols, stems_rests, notehead, clefs_keys
    )
