import numpy as np
from time import perf_counter

from homr.type_definitions import NDArray
from homr.inference_engine.onnx_model import OnnxModel
from homr.inference_engine.tflite_model import TensorFlowModel


class EncoderDual:
    def __init__(self, cnn_path: str, transformer_path) -> None:
        """
        Special dual encoder splitting the CNN and Transformer part of the
        Encoder for better performance on android (LiteRT is faster for CNNs
        while Onnx is faster for transformers)
        """
        self.cnn_encoder = TensorFlowModel(cnn_path)
        self.transformer_encoder = OnnxModel(transformer_path)

    def generate(self, x: NDArray) -> NDArray:
        t0 = perf_counter()
        out = self.cnn_encoder.run(x)
        t1 = perf_counter()

        embeddings = np.transpose(out, (0, 3, 1, 2))

        input_dict = {"input": embeddings.astype(np.float32)}
        output_dict = {"output": [1, 641, 312]}
        output = self.transformer_encoder.run(input_dict, output_dict)
        print(f"Inference time CNN part of Encoder: {round(t1 - t0, 3)}s")
        print(
            f"Inference time Transformer part of Encoder: {round(perf_counter() - t1, 3)}s"
        )
        return output[0]
