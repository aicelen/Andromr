import numpy as np
from time import perf_counter

from homr.type_definitions import NDArray
from homr.inference_engine import TensorFlowModel
from homr.transformer.configs import default_config

from globals import appdata

encoder: TensorFlowModel | None = None


class Encoder:
    def __init__(self) -> None:
        """
        Enocder using only one .tflite file.
        """
        global encoder
        if encoder is None or encoder.num_threads != appdata.threads:
            encoder = TensorFlowModel(default_config.filepaths.encoder_path)

        self.encoder = encoder

    def generate(self, x: NDArray) -> NDArray:
        t0 = perf_counter()
        out = self.encoder.run(x, (1, 1280, 512))
        t1 = perf_counter()

        print(f"Inference time of Encoder: {round(t1 - t0, 3)}s")

        return out.astype(np.float32)
