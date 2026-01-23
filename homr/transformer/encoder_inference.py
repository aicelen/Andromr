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

        preload_encoder(appdata.threads, appdata.gpu)

        if not encoder.loaded or appdata.settings_changed:
            encoder.load()

        self.encoder = encoder

    def generate(self, x: NDArray) -> NDArray:
        t0 = perf_counter()
        out = self.encoder.run(x)
        t1 = perf_counter()

        print(f"Inference time of Encoder: {round(t1 - t0, 3)}s")

        return out.astype(np.float32)


def preload_encoder(num_threads: int, use_gpu: bool) -> TensorFlowModel:
    """
    Load the CNN part of the encoder

    :param num_threads: Number of threads
    :type num_threads: int
    :param use_gpu: Use GPU for inference
    :type use_gpu: bool
    :return: The loaded TensorflowModel
    :rtype: TensorFlowModel
    """
    global encoder
    if encoder is None or appdata.settings_changed:
        encoder = TensorFlowModel(
            default_config.filepaths.encoder,
            num_threads=2,
            use_gpu=False,
            precision_loss=False,
        )
