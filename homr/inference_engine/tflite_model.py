"""
Copied from teticio's kivy-tensorflowlite-helloworld:
https://github.com/teticio/kivy-tensorflow-helloworld/blob/main/model.py
"""

import numpy as np
from kivy.utils import platform
import hashlib
import os

if platform == "android":
    from jnius import autoclass  # type: ignore

    PythonActivity = autoclass("org.kivy.android.PythonActivity")
    Context = autoclass("android.content.Context")

    File = autoclass("java.io.File")
    Interpreter = autoclass("org.tensorflow.lite.Interpreter")
    InterpreterOptions = autoclass("org.tensorflow.lite.Interpreter$Options")
    Tensor = autoclass("org.tensorflow.lite.Tensor")
    DataType = autoclass("org.tensorflow.lite.DataType")
    TensorBuffer = autoclass("org.tensorflow.lite.support.tensorbuffer.TensorBuffer")
    ByteBuffer = autoclass("java.nio.ByteBuffer")
    GpuDelegate = autoclass("org.tensorflow.lite.gpu.GpuDelegate")
    GpuDelegateOptions = autoclass("org.tensorflow.lite.gpu.GpuDelegate$Options")
    CompatibilityList = autoclass("org.tensorflow.lite.gpu.CompatibilityList")

    InterpreterApiOptions = autoclass("org.tensorflow.lite.InterpreterApi$Options")
    Delegate = autoclass("org.tensorflow.lite.Delegate")

    class TensorFlowModel:
        """
        Cross platform inference of .tflite models

        :param model_path: Path to the .tflite model
        :type model_path: str
        :param num_threads: Number of threads to use (CPU only)
        :type num_threads: int
        :param use_gpu: Use GPU acceleration
        :type use_gpu: bool
        :param precisionLoss: Use fp16 calculations to speed up
                              inference (only works with use_gpu=True)
        :type precisionLoss: bool
        """

        def __init__(
            self,
            model_filename: str,
            num_threads: int = 1,
            use_gpu: bool = True,
            precision_loss: bool = True,
            sustained_speed: bool = False,
        ):
            self.loaded = False
            self.model_filename = model_filename
            self.options = InterpreterOptions()
            self.compatList = CompatibilityList()
            if use_gpu and self.compatList.isDelegateSupportedOnThisDevice():
                # Delegate Options
                delegate_options = self.compatList.getBestOptionsForThisDevice()
                delegate_options = delegate_options.setPrecisionLossAllowed(precision_loss)
                delegate_options = delegate_options.setInferencePreference(
                    1 if sustained_speed else 0
                )

                if os.path.exists(self.model_filename):
                    serialization_dir = self._get_serialization_dir()
                    model_token = self._compute_model_token(self.model_filename)

                    delegate_options = delegate_options.setSerializationParams(
                        serialization_dir,
                        model_token,
                    )

                gpu_delegate = GpuDelegate(delegate_options)
                self.options.addDelegate(gpu_delegate)
                print("Set GPU")
            else:
                self.options.setNumThreads(num_threads)
                self.options.setUseXNNPACK(True)
                print("Set CPU")

        def load(self):
            """
            Loads the model from the path given in the constructor
            """
            model = File(self.model_filename)
            self.interpreter = Interpreter(model, self.options)
            self.allocate_tensors()
            self.loaded = True

        def _compute_model_token(self, model_path: str) -> str:
            """
            Computes the token of the model
            """
            sha = hashlib.sha256()
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha.update(chunk)
            return sha.hexdigest()

        def _get_serialization_dir(self) -> str:
            """
            Find a directory for GPU delegate serialization
            """
            activity = PythonActivity.mActivity
            code_cache_dir = activity.getCodeCacheDir()
            ser_dir = File(code_cache_dir, "tflite_gpu_serialization")
            if not ser_dir.exists():
                ser_dir.mkdirs()
            return ser_dir.getAbsolutePath()

        def allocate_tensors(self):
            self.interpreter.allocateTensors()
            self.input_shape = self.interpreter.getInputTensor(0).shape()
            self.output_shape = self.interpreter.getOutputTensor(0).shape()
            self.output_type = self.interpreter.getOutputTensor(0).dataType()

        def get_input_shape(self):
            return self.input_shape

        def resize_input(self, shape):
            if self.input_shape != shape:
                self.interpreter.resizeInput(0, shape)
                self.allocate_tensors()

        def run(self, x: np.ndarray):
            """
            Docstring for run

            :param x: Input array
            """
            # assumes one input and one output for now
            input = ByteBuffer.wrap(x.tobytes())
            output = TensorBuffer.createFixedSize(self.output_shape, self.output_type)
            self.interpreter.run(input, output.getBuffer().rewind())
            return np.reshape(np.array(output.getFloatArray()), self.output_shape)

else:
    if platform == "win":
        import tensorflow as tf  # type: ignore

        Interpreter = tf.lite.Interpreter

    else:
        # ai-edege-litert is only available on Linux/WSL and MacOS
        from ai_edge_litert.interpreter import Interpreter

    class TensorFlowModel:
        """
        Cross platform inference of .tflite models

        :param model_path: Path to the .tflite model
        :param num_threads: Number of threads to use (CPU only)
        :param use_gpu: Use GPU acceleration
        :param precisionLoss: Use fp16 calculations to speed up
                              inference (only works with use_gpu=True)
        """

        def __init__(
            self,
            model_filename: str,
            num_threads: int = 1,
            use_gpu: bool = True,
            precision_loss: bool = True,
            sustained_speed: bool = False,
        ):
            self.model_path = model_filename
            self.num_threads = num_threads
            self.loaded = False

        def load(self):
            self.interpreter = Interpreter(self.model_path, num_threads=self.num_threads)
            self.interpreter.allocate_tensors()
            self.loaded = True

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
