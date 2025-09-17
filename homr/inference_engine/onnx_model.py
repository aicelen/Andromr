"""
Inference of .onnx models for kivy android apps using native Java APIs.
Made by aicelen 2025 released under MIT license.
"""

import numpy as np
from time import perf_counter
from kivy.utils import platform


if platform == 'android':
    from jnius import autoclass

    # Import Java classes
    OrtSession = autoclass('ai.onnxruntime.OrtSession')
    OrtSessionOptions = autoclass('ai.onnxruntime.OrtSession$SessionOptions')
    OrtEnvironment = autoclass('ai.onnxruntime.OrtEnvironment')
    OnnxTensor = autoclass('ai.onnxruntime.OnnxTensor')
    ByteBuffer = autoclass('java.nio.ByteBuffer')
    FloatBuffer = autoclass('java.nio.FloatBuffer')
    HashMap = autoclass('java.util.HashMap')
    OnnxJavaType = autoclass('ai.onnxruntime.OnnxJavaType')
    Array = autoclass('java.util.Arrays')
    ByteOrder = autoclass('java.nio.ByteOrder')

    class OnnxModel:
        def __init__(
                self, 
                model_path: str, 
                num_threads: int = None, 
                use_nnapi: bool = False, 
                use_xnnpack: bool = False
            ):
            """
            Inference class of .onnx models for kivy apps on android using native Java APIs.

            Parameters
            ----------
            model_path: str
                Path to the .onnx model you want to run inference on.
            num_threads: int = None
                How many threads the model inference should use. Usually best performance 
                is achieved by using all big cores of the SOC. Defaults to 1.
            use_nnapi: bool = False
                Neural Networks API can provide significant speed ups although it is 
                often not compatible with the models and is not further developed.
            use_xnnpack: bool = False
                Should speed up inference time for some models.
            """
            self.env = OrtEnvironment.getEnvironment()
            so = OrtSessionOptions()

            if use_nnapi:
                so.addNnapi()

            if use_xnnpack:
                xnnpack_map = HashMap()
                xnnpack_map.put("intra_op_num_threads", str(num_threads or 2))
                so.addXnnpack(xnnpack_map)

            elif num_threads is not None:
                so.setIntraOpNumThreads(num_threads)
                so.setInterOpNumThreads(1)
        
            self.session = self.env.createSession(model_path, so)

        def run(self, inputs: dict, outputs: dict) -> dict:
            """
            Run inference on a given set of inputs.

            Parameters
            ----------
            inputs : dict
                Dictionary representing your input data with the format {your_input_name: np.array(your_data)}
            outputs: dict
                Dictionary for the output shape with the format {your_output_name: [shape]}

            Returns
            -------
            dict
                Outputs of the model
            """
            jmap = HashMap()
            for name, value in inputs.items():
                arr = np.ascontiguousarray(value)
                shape = list(arr.shape)

                if arr.dtype == np.float32:
                    flat = arr.ravel()
                    buffer_bytes = flat.tobytes()
                    java_byte_buffer = ByteBuffer.wrap(buffer_bytes)
                    java_byte_buffer.order(ByteOrder.nativeOrder())
                    float_buffer = java_byte_buffer.asFloatBuffer()
                    tensor = OnnxTensor.createTensor(self.env, float_buffer, shape)

                elif arr.dtype == np.int64:
                    flat = arr.ravel().astype(np.int64)
                    buffer_bytes = flat.tobytes()
                    java_byte_buffer = ByteBuffer.wrap(buffer_bytes)
                    java_byte_buffer.order(ByteOrder.nativeOrder())
                    long_buffer = java_byte_buffer.asLongBuffer()
                    tensor = OnnxTensor.createTensor(self.env, long_buffer, shape)

                else:
                    raise TypeError(f"Unsupported dtype of input array: {arr.dtype}")
                jmap.put(name, tensor)

            t0 = perf_counter()
            results = self.session.run(jmap)
            print(f"Raw inference time: {perf_counter() - t0:.4f}s")

            output_list = []
            for out_name, shape in outputs.items():
                tensor_obj = results.get(out_name).get()
                bytebuffer = bytes(tensor_obj.getByteBuffer().array())
                numpy_array = np.frombuffer(bytebuffer, dtype=np.float32)

                # Reshape and add the output data to the dict.
                output_list.append(numpy_array.reshape(*shape))
                tensor_obj.close()

            results.close()
            return output_list

        def close_session(self):
            """
            Frees up memory. Should be called when a model isn't used anymore
            """
            self.session.close()
else:
    import onnxruntime as ort
    class OnnxModel:
        def __init__(
            self, 
            model_path: str, 
            num_threads: int = None, 
            use_nnapi: bool = False, 
            use_xnnpack: bool = False
        ):
            self.model = ort.InferenceSession(model_path)
        
        def run(self, inputs: dict, outputs: dict = None) -> dict:
            result = self.model.run(output_names=list(outputs.keys()), input_feed=inputs)
            return result
        
        def close_session(self):
            print('Not implemented')
