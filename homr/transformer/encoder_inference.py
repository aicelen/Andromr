import numpy as np
from time import perf_counter

from homr.type_definitions import NDArray
from homr.inference_engine import OnnxModel
from homr.inference_engine import TensorFlowModel
from homr.transformer.configs import default_config

from globals import appdata

cnn_encoder = None
transformer_encoder = None

class EncoderDual:
    def __init__(self) -> None:
        """
        Special dual encoder splitting the CNN and Transformer part of the
        Encoder for better performance on android (LiteRT is faster for CNNs
        while Onnx is faster for transformers).
        """
        global cnn_encoder, transformer_encoder

        load_cnn_encoder()
        load_transformer_encoder()

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
        print(
            f"Inference time Transformer part of Encoder: {round(perf_counter() - t1, 3)}s"
        )
        return output[0]

def load_cnn_encoder(num_threads: int = appdata.threads, use_gpu: bool = False) -> TensorFlowModel:
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
    if cnn_encoder is None:
        cnn_encoder = TensorFlowModel(default_config.filepaths.encoder_cnn_path_tflite, num_threads, use_gpu, False)

def load_transformer_encoder(num_threads: int = appdata.threads) -> OnnxModel:
    """
    Load the transformer part of the encoder.
    
    :param num_threads: Nuber of threads the model uses
    :type num_threads: int
    :return: The loaded OnnxModel
    :rtype: OnnxModel
    """
    global transformer_encoder
    if transformer_encoder is None:
        transformer_encoder = OnnxModel(default_config.filepaths.encoder_transformer_path, num_threads=num_threads)
