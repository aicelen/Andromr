from time import perf_counter, time
from os import cpu_count

import numpy as np

from homr.inference_engine import OnnxModel, TensorFlowModel
from homr.transformer.configs import default_config
from homr.segmentation.config import segnet_path_tflite

class Benchmark():
    def __init__(self):
        self.check_threads = [1, 2, 4, 6, cpu_count()]
        self.cycles = 10

    def warmup(self, warmup_duration=30):
        warmup_model = TensorFlowModel(segnet_path_tflite, num_threads=cpu_count())
        start_time = time()
        input_array = np.random.random((1, 320, 320, 3)).astype(np.float32)
        while time() - start_time < warmup_duration:
            warmup_model.run(input_array)

    def benchmark_tflite(self, threshold=0):
        best_result = (0, np.inf)
        input_segnet = np.random.random((1, 320, 320, 3)).astype(np.float32)
        input_encoder = np.random.random((1, 128, 1280, 1)).astype(np.float32)
        for threads in self.check_threads:
            segnet = TensorFlowModel(segnet_path_tflite, num_threads=threads)
            encoder = TensorFlowModel(default_config.filepaths.encoder_cnn_path_tflite, num_threads=threads)
            cur_time = 0

            for i in range(self.cycles):
                t0 = perf_counter()
                segnet.run(input_segnet)
                encoder.run(input_encoder)
                cur_time += perf_counter() - t0

            print(threads, cur_time)

            if best_result[1] > cur_time + threshold:
                best_result = (threads, cur_time)

        return best_result[0]

    def benchmark_onnx(self, threshold=0):
        best_result = (0, np.inf)
        input_encoder = {
            "input": np.random.random((1, 312, 8, 80)).astype(np.float32)
        }

        output_encoder = {
            "output": [1, 641, 312]
        }

        input_decoder = {
            "rhythms": np.random.randint(0, 4, (1, 20)).astype(np.int64),
            "pitchs": np.random.randint(0, 4, (1, 20)).astype(np.int64),
            "lifts": np.random.randint(0, 4, (1, 20)).astype(np.int64),
            "context": np.random.rand(1, 641, 312).astype(np.float32),
        }
        outputs_decoder = {
                "out_rhythms": [1, 20, 93],
                "out_pitchs": [1, 20, 71],
                "out_lifts": [1, 20, 5],
        }

        for threads in self.check_threads:
            encoder = OnnxModel(default_config.filepaths.encoder_transformer_path, num_threads=threads)
            decoder = OnnxModel(default_config.filepaths.decoder_path, num_threads=threads)
            cur_time = 0

            for i in range(self.cycles):
                t0 = perf_counter()
                encoder.run(input_encoder, output_encoder)
                for k in range(5):
                    # Decoder is way faster
                    decoder.run(input_decoder, outputs_decoder)
                cur_time += perf_counter() - t0

            print(threads, cur_time)

            if best_result[1] > cur_time + threshold:
                best_result = (threads, cur_time)

        return best_result[0]


    def run(self):
        self.warmup()
        tflite = self.benchmark_tflite(threshold=0.1)
        onnx = self.benchmark_onnx(threshold=0.05)
        print(tflite, onnx)

if __name__ == '__main__':
    test = Benchmark()
    test.run()