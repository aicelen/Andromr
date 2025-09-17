import onnxruntime as ort
from time import perf_counter

from homr.type_definitions import NDArray
from homr.inference_engine.onnx_model import OnnxModel
from homr.inference_engine.tflite_model import TensorFlowModel

class EncoderDual:
    def __init__(self, cnn_path: str, transformer_path, use_gpu: bool) -> None:
        """
        Special dual encoder splitting the CNN and Transformer part of the 
        Encoder for better performance on android (LiteRT is faster for CNNs 
        while Onnx is faster for transformers)
        """
        if use_gpu:
            try:
                self.cnn_encoder = ort.InferenceSession(cnn_path, providers=["CUDAExecutionProvider"])
            except Exception:
                self.cnn_encoder = ort.InferenceSession(cnn_path)

        else:
            self.cnn_encoder = ort.InferenceSession(cnn_path)

        
        self.transformer_encoder = OnnxModel(transformer_path)

        self.cnn_input_name = self.cnn_encoder.get_inputs()[0].name
        self.cnn_output_name = self.cnn_encoder.get_outputs()[0].name

    def generate(self, x: NDArray) -> NDArray:
        t0 = perf_counter()
        embeddings = self.cnn_encoder.run([self.cnn_output_name], {self.cnn_input_name: x})
        t1 = perf_counter()

        input_dict = {"input": embeddings[0]}
        output_dict = {"output": [1,641,312]}
        output = self.transformer_encoder.run(input_dict, output_dict)
        print(f"Inference time CNN part of Encoder: {round(perf_counter() - t0, 3)}s")
        print(f"Inference time Transformer part of Encoder: {round(perf_counter() - t1, 3)}s")
        return output[0]