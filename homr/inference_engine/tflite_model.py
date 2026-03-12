"""
Based on https://github.com/aicelen/Kivy-LiteRT-Next
"""

from kivy.utils import platform
import numpy as np
from time import perf_counter

from globals import appdata
from homr.simple_logging import eprint

if platform == "android":
    from jnius import autoclass  # type: ignore

    # Acquire Android context via PythonActivity
    PythonActivity = autoclass("org.kivy.android.PythonActivity")
    context = PythonActivity.mActivity

    # Import Lite RT Next components (adjust package path as needed)
    CompiledModel = autoclass("com.google.ai.edge.litert.CompiledModel")
    Accelerator = autoclass("com.google.ai.edge.litert.Accelerator")
    Options = autoclass("com.google.ai.edge.litert.CompiledModel$Options")
    CpuOptions = autoclass("com.google.ai.edge.litert.CompiledModel$CpuOptions")
    HashSet = autoclass("java.util.HashSet")
    Integer = autoclass('java.lang.Integer')
    # buildozer is sometimes cutting unused classes away
    # but Tensorbuffer is used run_inference()
    # importing stops buildozer from cutting it away
    TensorBuffer = autoclass("com.google.ai.edge.litert.TensorBuffer")

    class TensorFlowModel:
        """
        Class for inference of a .tflite model using LiteRT Compiled Model API (version 2.1.1)
        Args:
            model_name(str): Name of the model for inference;
            use_gpu(bool): If True uses GPU acceleration, FP16 model only
        """

        def __init__(
            self,
            model_filename: str
        ):
            accelerator_set = HashSet()
            self.use_gpu = appdata.gpu
            self.num_threads = appdata.threads
            eprint("Using GPU acceleration" if self.use_gpu else f"Using CPU with {self.num_threads} threads.")

            if self.use_gpu:
                accelerator_set.add(Accelerator.GPU)
                opts = Options(accelerator_set)
            else:
                accelerator_set.add(Accelerator.CPU)
                opts = Options(accelerator_set)
                cpu_opts = CpuOptions(Integer(self.num_threads), None, None)
                opts.setCpuOptions(cpu_opts)

            self.model = CompiledModel.create(model_filename, opts)
            self.input_buffers = self.model.createInputBuffers()
            self.output_buffers = self.model.createOutputBuffers()

        def run(self, input_data, output_shape: tuple, int64: bool = False):
            """
            Method of TensorFlowModel class performing inference of chosen model
            Args:
                input_data(np.array): Input data for the model
            Returns:
                result: Ouput of model
            """
            # Convert numpy array to flattened list
            if hasattr(input_data, "tolist"):
                input_data = input_data.astype("float32").flatten().tolist()

            # fill buffer with input data
            buf_in = self.input_buffers.get(0)
            buf_in.writeFloat(input_data)

            # run inference
            t0 = perf_counter()
            self.model.run(self.input_buffers, self.output_buffers)
            print(perf_counter() - t0)

            # read output from the first output tensor
            buf_out = self.output_buffers.get(0)
            if int64:
                result = buf_out.readLong()
                return np.array(result, dtype=np.int64).reshape(output_shape)
            else:
                result = buf_out.readFloat()
                return np.array(result, dtype=np.float32).reshape(output_shape)

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
        :param use_gpu: Use GPU acceleration; Not Implemented
        """

        def __init__(
            self,
            model_filename: str,
        ):
            self.use_gpu = appdata.gpu
            self.num_threads = appdata.threads
            eprint("GPU acceleration is only supported on Android." if self.use_gpu else f"Using CPU with {self.num_threads} threads.")

            self.model_path = model_filename
            self.interpreter = Interpreter(self.model_path)
            self.interpreter.allocate_tensors()

        def resize_input(self, shape):
            if list(self.get_input_shape()) != shape:
                self.interpreter.resize_tensor_input(0, shape)
                self.interpreter.allocate_tensors()

        def get_input_shape(self):
            return self.interpreter.get_input_details()[0]["shape"]

        def run(self, x, output_shape: tuple, int64: bool = False):
            # assumes one input and one output for now
            self.interpreter.set_tensor(self.interpreter.get_input_details()[0]["index"], x)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.interpreter.get_output_details()[0]["index"])
