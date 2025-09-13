import onnxruntime as ort
from homr.type_definitions import NDArray
from time import perf_counter

class Encoder:
    def __init__(self, path: str, use_gpu: bool) -> None:
        if use_gpu:
            try:
                self.encoder = ort.InferenceSession(path, providers=["CUDAExecutionProvider"])
            except Exception:
                self.encoder = ort.InferenceSession(path)

        else:
            self.encoder = ort.InferenceSession(path)

        self.input_name = self.encoder.get_inputs()[0].name
        self.output_name = self.encoder.get_outputs()[0].name

    def generate(self, x: NDArray) -> NDArray:
        output = self.encoder.run([self.output_name], {self.input_name: x})
        return output[0]

class EncoderDual:
    def __init__(self, cnn_path: str, transformer_path, use_gpu: bool) -> None:
        if use_gpu:
            try:
                self.cnn_encoder = ort.InferenceSession(cnn_path, providers=["CUDAExecutionProvider"])
            except Exception:
                self.cnn_encoder = ort.InferenceSession(cnn_path)

        else:
            self.cnn_encoder = ort.InferenceSession(cnn_path)

        
        if use_gpu:
            try:
                self.transformer_encoder = ort.InferenceSession(transformer_path, providers=["CUDAExecutionProvider"])
            except Exception:
                self.transformer_encoder = ort.InferenceSession(transformer_path)

        else:
            self.transformer_encoder = ort.InferenceSession(transformer_path)

        self.cnn_input_name = self.cnn_encoder.get_inputs()[0].name
        self.cnn_output_name = self.cnn_encoder.get_outputs()[0].name
        self.transformer_input_name = self.transformer_encoder.get_inputs()[0].name
        self.transformer_output_name = self.transformer_encoder.get_outputs()[0].name

    def generate(self, x: NDArray) -> NDArray:
        t0 = perf_counter()
        embeddings = self.cnn_encoder.run([self.cnn_output_name], {self.cnn_input_name: x})
        t1 = perf_counter()
        output = self.transformer_encoder.run([self.cnn_output_name], {self.cnn_input_name: embeddings[0]})
        print(f"Inference time CNN part of Encoder: {round(perf_counter() - t0, 3)}s")
        print(f"Inference time Transformer part of Encoder: {round(perf_counter() - t1, 3)}s")
        return output[0]