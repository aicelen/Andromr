
from typing import Tuple

import cv2
import numpy as np
from numpy import ndarray


def get_kernel(kernel: Tuple[int, int]) -> ndarray:
    if isinstance(kernel, tuple):
        # It's kernel shape
        kernel = np.ones(kernel, dtype=np.uint8)  # type: ignore
    return kernel  # type: ignore


def morph_open(img: ndarray, kernel: Tuple[int, int]) -> ndarray:
    ker = get_kernel(kernel)
    return cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_OPEN, ker)


def morph_close(img: ndarray, kernel: Tuple[int, int]) -> ndarray:
    ker = get_kernel(kernel)
    return cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_CLOSE, ker)


def morph_hit_miss(img, kernel):
    ker = get_kernel(kernel)
    return cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_HITMISS, ker)

