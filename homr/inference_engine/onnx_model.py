import onnxruntime as ort


class OnnxModel:
    def __init__(self, model_path: str, num_threads: int = 0):
        self.model_path = model_path
        self.num_threads = num_threads
        self.xnnpack = False
        self.session = None
        self.session_options = ort.SessionOptions()
        self.session_options.intra_op_num_threads = 0

    def load(self):
        self.session = ort.InferenceSession(self.model_path, self.session_options)

    def run(self, inputs: dict, outputs: dict) -> dict:
        return self.session.run(output_names=list(outputs), input_feed=inputs)
