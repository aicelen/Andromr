"""
Copied from teticio's kivy-tensorflowlite-helloworld:
https://github.com/teticio/kivy-tensorflow-helloworld/blob/main/model.py
"""

import numpy as np
from kivy.utils import platform
from globals import appdata
from time import perf_counter

if platform == "android":
    from jnius import autoclass  # type: ignore

    File = autoclass("java.io.File")
    Interpreter = autoclass("org.tensorflow.lite.Interpreter")
    InterpreterOptions = autoclass("org.tensorflow.lite.Interpreter$Options")
    Tensor = autoclass("org.tensorflow.lite.Tensor")
    DataType = autoclass("org.tensorflow.lite.DataType")
    TensorBuffer = autoclass("org.tensorflow.lite.support.tensorbuffer.TensorBuffer")
    ByteBuffer = autoclass("java.nio.ByteBuffer")
    ByteOrder = autoclass("java.nio.ByteOrder")
    GpuDelegate = autoclass('org.tensorflow.lite.gpu.GpuDelegate')
    CompatibilityList = autoclass("org.tensorflow.lite.gpu.CompatibilityList")
    
    # dummy import so buildozer isn't cutting it away since it's used by options.setNumThreads
    InterpreterApiOptions = autoclass("org.tensorflow.lite.InterpreterApi$Options")
    Delegate = autoclass("org.tensorflow.lite.Delegate")

    print('Imported everything')

    class TensorFlowModel:
        def __init__(self, model_filename, num_threads=None):
            if num_threads is None:
                num_threads = appdata.threads

            model = File(model_filename)
            options = InterpreterOptions()

            # GPU Delegate Setup
            if True:
                print('Initializing GPU Delegate...')
                self.gpu_delegate = GpuDelegate()
                options.addDelegate(self.gpu_delegate)
            else:
                options.setNumThreads(num_threads)
                options.setUseXNNPACK(True)
            
            self.interpreter = Interpreter(model, options)
            self.interpreter.allocateTensors()

            # Cache shapes and types
            self.input_shape = self.interpreter.getInputTensor(0).shape()
            self.output_shape = self.interpreter.getOutputTensor(0).shape()
            
            # Calculate buffer sizes (assuming float32 = 4 bytes)
            # numpy shape to total elements
            self.input_size_bytes = np.prod(self.input_shape) * 4
            self.output_size_bytes = np.prod(self.output_shape) * 4
            
            # OPTIMIZATION 1: Pre-allocate Direct ByteBuffers
            # Direct buffers are required for zero-copy passing to GPU delegates
            self.input_buffer = ByteBuffer.allocateDirect(int(self.input_size_bytes))
            self.input_buffer.order(ByteOrder.nativeOrder())
            
            self.output_buffer = ByteBuffer.allocateDirect(int(self.output_size_bytes))
            self.output_buffer.order(ByteOrder.nativeOrder())

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
            # OPTIMIZATION 2: Ensure Contiguity
            # If x is a slice (from main app), this makes it contiguous fast.
            # If x is already contiguous, this is near-instant.
            if not x.flags['C_CONTIGUOUS']:
                x = np.ascontiguousarray(x)

            # 1. Fill Input Buffer
            self.input_buffer.rewind()
            # x.tobytes() is fast on contiguous arrays
            self.input_buffer.put(x.tobytes()) 
            
            # 2. Run Inference
            self.input_buffer.rewind()
            self.output_buffer.rewind()
            self.interpreter.run(self.input_buffer, self.output_buffer)
            
            # 3. Fast Output Reading
            # Avoid getFloatArray() -> List -> Numpy (SLOW)
            # Instead, read raw bytes into a Python bytearray
            self.output_buffer.rewind()
            out_bytes = bytearray(int(self.output_size_bytes))
            self.output_buffer.get(out_bytes)
            
            # Zero-copy conversion from bytes to numpy
            return np.frombuffer(out_bytes, dtype=np.float32).reshape(self.output_shape)
else:
    if platform == 'win':
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    else:
        # ai-edege-litert is only available on Linux/WSL and MacOS 
        from ai_edge_litert.interpreter import Interpreter

    class TensorFlowModel:
        def __init__(self, model_filename, num_threads=8):
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
