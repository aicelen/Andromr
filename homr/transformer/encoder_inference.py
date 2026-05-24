from time import perf_counter

import numpy as np

from homr.transformer.configs import default_config
from homr.type_definitions import NDArray

encoder = None

from homr.inference_engine import TensorFlowModel


class Encoder:
    def __init__(self) -> None:
        """
        Enocder using only one .tflite file.
        """
        global encoder
        self.encoder = TensorFlowModel(default_config.filepaths.encoder_path)

    def generate(self, x: NDArray) -> NDArray:
        t0 = perf_counter()
        out = self.encoder.run(x)
        t1 = perf_counter()

        print(f"Inference time of Encoder: {round(t1 - t0, 3)}s")

        return out.astype(np.float32)
