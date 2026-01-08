import numpy as np
from time import perf_counter

from homr.type_definitions import NDArray
from homr.inference_engine import OnnxModel
from homr.inference_engine import TensorFlowModel
from homr.transformer.configs import default_config

from globals import appdata

cnn_encoder: TensorFlowModel | None = None
transformer_encoder: OnnxModel | None = None


class EncoderDual:
    def __init__(self) -> None:
        """
        Special dual encoder splitting the CNN and Transformer part of the
        Encoder for better performance on android (LiteRT is faster for CNNs
        while Onnx is faster for transformers).
        """
        global cnn_encoder, transformer_encoder

        preload_cnn_encoder(appdata.threads, appdata.gpu)
        preload_transformer_encoder(appdata.threads)

        if not cnn_encoder.loaded or appdata.settings_changed:
            cnn_encoder.load()
        
        if appdata.settings_changed or transformer_encoder.session is None:
            transformer_encoder.load()

        self.cnn_encoder = cnn_encoder
        self.transformer_encoder = transformer_encoder

    def generate(self, x: NDArray) -> NDArray:
        t0 = perf_counter()
        out = self.cnn_encoder.run(x)
        t1 = perf_counter()

        embeddings = np.transpose(out, (0, 3, 1, 2))

        input_dict = {"input": embeddings.astype(np.float32)}
        output_dict = {"output": [1, 1281, 312]}
        output = self.transformer_encoder.run(input_dict, output_dict)
        print(f"Inference time CNN part of Encoder: {round(t1 - t0, 3)}s")
        print(f"Inference time Transformer part of Encoder: {round(perf_counter() - t1, 3)}s")
        return output[0]


def preload_cnn_encoder(num_threads: int, use_gpu: bool) -> TensorFlowModel:
    """
    Load the CNN part of the encoder

    :param num_threads: Number of threads
    :type num_threads: int
    :param use_gpu: Use GPU for inference
    :type use_gpu: bool
    :return: The loaded TensorflowModel
    :rtype: TensorFlowModel
    """
    global cnn_encoder
    if cnn_encoder is None or appdata.settings_changed:
        cnn_encoder = TensorFlowModel(
            default_config.filepaths.encoder_cnn_path_tflite, num_threads=num_threads, use_gpu=use_gpu, precision_loss=False
        )


def preload_transformer_encoder(num_threads: int) -> OnnxModel:
    """
    Load the transformer part of the encoder.

    :param num_threads: Nuber of threads the model uses
    :type num_threads: int
    :return: The loaded OnnxModel
    :rtype: OnnxModel
    """
    global transformer_encoder
    if transformer_encoder is None or appdata.settings_changed:
        transformer_encoder = OnnxModel(
            default_config.filepaths.encoder_transformer_path, num_threads=num_threads
        )
