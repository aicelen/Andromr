from kivy.utils import platform

from globals import appdata
from homr.simple_logging import eprint

if platform == "win":
    import tensorflow as tf  # type: ignore

    Interpreter = tf.lite.Interpreter

else:
    # ai-edege-litert is only available on Linux/WSL and MacOS
    from ai_edge_litert.interpreter import Interpreter


class TensorFlowModel:
    """
    Cross platform inference of .tflite models

    :param model_filename: Path to the .tflite model
    """

    def __init__(
        self,
        model_filename: str,
    ):
        self.num_threads = appdata.threads

        self.model_path = model_filename
        self.interpreter = Interpreter(self.model_path)
        self.interpreter.allocate_tensors()

    def resize_input(self, shape):
        if list(self.get_input_shape()) != shape:
            self.interpreter.resize_tensor_input(0, shape)
            self.interpreter.allocate_tensors()

    def get_input_shape(self):
        return self.interpreter.get_input_details()[0]["shape"]

    def run(self, x):
        # assumes one input and one output for now
        self.interpreter.set_tensor(self.interpreter.get_input_details()[0]["index"], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.interpreter.get_output_details()[0]["index"])
