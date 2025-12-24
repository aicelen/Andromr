"""
Copied from teticio's kivy-tensorflowlite-helloworld:
https://github.com/teticio/kivy-tensorflow-helloworld/blob/main/model.py
"""

import numpy as np
from kivy.utils import platform
from globals import appdata

if platform == "android":
    from jnius import autoclass  # type: ignore

    File = autoclass("java.io.File")
    Interpreter = autoclass("org.tensorflow.lite.Interpreter")
    InterpreterOptions = autoclass("org.tensorflow.lite.Interpreter$Options")
    Tensor = autoclass("org.tensorflow.lite.Tensor")
    DataType = autoclass("org.tensorflow.lite.DataType")
    TensorBuffer = autoclass("org.tensorflow.lite.support.tensorbuffer.TensorBuffer")
    ByteBuffer = autoclass("java.nio.ByteBuffer")
    GpuDelegate = autoclass('org.tensorflow.lite.gpu.GpuDelegate')
    GpuDelegateOptions = autoclass('org.tensorflow.lite.gpu.GpuDelegate$Options')
    CompatibilityList = autoclass("org.tensorflow.lite.gpu.CompatibilityList")

    InterpreterApiOptions = autoclass("org.tensorflow.lite.InterpreterApi$Options")
    Delegate = autoclass("org.tensorflow.lite.Delegate")

    class TensorFlowModel:
        def __init__(self, model_filename, num_threads=None, use_gpu=False, precisionLoss=True):
            model = File(model_filename)
            options = InterpreterOptions()

            if use_gpu:
                gpu_options = GpuDelegateOptions().setQuantizedModelsAllowed(True).setPrecisionLossAllowed(precisionLoss)
                gpu_delegate = GpuDelegate(gpu_options)
                options.addDelegate(gpu_delegate)
                print('set gpu')
            else:
                options.setNumThreads(num_threads)
                options.setUseXNNPACK(True)
            self.interpreter = Interpreter(model, options)
            self.allocate_tensors()

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

        def run(self, x):
            # assumes one input and one output for now
            input = ByteBuffer.wrap(x.tobytes())
            output = TensorBuffer.createFixedSize(self.output_shape, self.output_type)
            self.interpreter.run(input, output.getBuffer().rewind())
            return np.reshape(np.array(output.getFloatArray()), self.output_shape)

else:
    if platform == 'win':
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    else:
        # ai-edege-litert is only available on Linux/WSL and MacOS 
        from ai_edge_litert.interpreter import Interpreter

    class TensorFlowModel:
        def __init__(self, model_filename, num_threads=8, use_gpu=None, precisionLoss=None):
            self.interpreter = Interpreter(
                model_filename, num_threads=num_threads
            )
            self.interpreter.allocate_tensors()

        def resize_input(self, shape):
            if list(self.get_input_shape()) != shape:
                self.interpreter.resize_tensor_input(0, shape)
                self.interpreter.allocate_tensors()

        def get_input_shape(self):
            return self.interpreter.get_input_details()[0]["shape"]

        def run(self, x):
            # assumes one input and one output for now
            self.interpreter.set_tensor(
                self.interpreter.get_input_details()[0]["index"], x
            )
            self.interpreter.invoke()
            return self.interpreter.get_tensor(
                self.interpreter.get_output_details()[0]["index"]
            )
