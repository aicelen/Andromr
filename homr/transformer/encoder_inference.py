from time import perf_counter

import numpy as np
from kivy.utils import platform

from globals import appdata
from homr.transformer.configs import default_config
from homr.type_definitions import NDArray

encoder = None

if platform == "android":
    from jnius import autoclass  # type: ignore

    LiteRTModel = autoclass("com.aicelen.andromr.LiteRTModel")
    ByteBuffer = autoclass("java.nio.ByteBuffer")
    ByteOrder = autoclass("java.nio.ByteOrder")
else:
    from homr.inference_engine import TensorFlowModel

class Encoder:
    def __init__(self) -> None:
        """
        Enocder using only one .tflite file.
        """
        global encoder
        if platform == "android" and encoder is None:
            encoder = LiteRTModel()
            encoder.load(str(default_config.filepaths.encoder_path), appdata.threads)
        elif platform != "android" and (
            encoder is None or encoder.num_threads != appdata.threads
        ):
            encoder = TensorFlowModel(default_config.filepaths.encoder_path)

        self.encoder = encoder

    def generate(self, x: NDArray) -> NDArray:
        t0 = perf_counter()
        if platform == "android":
            out = self.inference_android_helper(x)
        else:
            out = self.encoder.run(x, (1, 1280, 512))
        t1 = perf_counter()

        print(f"Inference time of Encoder: {round(t1 - t0, 3)}s")

        return out.astype(np.float32)

    def inference_android_helper(self, image):
        global encoder
        image = np.ascontiguousarray(image)
        flat = image.ravel()
        buffer_bytes = flat.tobytes()
        java_byte_buffer = ByteBuffer.wrap(buffer_bytes)
        java_byte_buffer.order(ByteOrder.nativeOrder())
        float_buffer = java_byte_buffer.asFloatBuffer()

        result = encoder.runFloat(float_buffer)
        return np.array(result, dtype=np.float32).reshape((1, 1280, 512))
