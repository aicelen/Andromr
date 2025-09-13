import numpy as np
from kivy.utils import platform

if platform == 'android':
    from jnius import autoclass

    File = autoclass('java.io.File')
    Interpreter = autoclass('org.tensorflow.lite.Interpreter')
    InterpreterOptions = autoclass('org.tensorflow.lite.Interpreter$Options')
    InterpreterApiOptions = autoclass('org.tensorflow.lite.InterpreterApi$Options')
    Tensor = autoclass('org.tensorflow.lite.Tensor')
    DataType = autoclass('org.tensorflow.lite.DataType')
    TensorBuffer = autoclass('org.tensorflow.lite.support.tensorbuffer.TensorBuffer')
    ByteBuffer = autoclass('java.nio.ByteBuffer')

    class TensorFlowModel():
        def load(self, model_filename):
            model = File(model_filename)
            options = InterpreterOptions()
            #options.setNumThreads(2)
            self.interpreter = Interpreter(model, options)
            self.allocate_tensors()

        def allocate_tensors(self):
            self.interpreter.allocateTensors()
            self.input_shape = self.interpreter.getInputTensor(0).shape()
            self.output_shape = self.interpreter.getOutputTensor(0).shape()
            self.output_type = self.interpreter.getOutputTensor(0).dataType()

        def get_input_shape(self):
            return self.input_shape

        def get_output_shape(self):
            return self.output_shape

        def resize_input(self, shape):
            if self.input_shape != shape:
                self.interpreter.resizeInput(0, shape)
                self.allocate_tensors()

        def run_inference(self, x):
            # assumes one input and one output for now
            input = ByteBuffer.wrap(x.tobytes())
            output = TensorBuffer.createFixedSize(self.output_shape,
                                                  self.output_type)
            self.interpreter.run(input, output.getBuffer().rewind())
            return np.reshape(np.array(output.getFloatArray()),
                              self.output_shape)
