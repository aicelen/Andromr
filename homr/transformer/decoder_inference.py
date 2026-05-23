from typing import Any

import numpy as np
import onnxruntime as ort

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray


class ScoreDecoder:
    def __init__(
        self,
        transformer: ort.InferenceSession,
        fp16: bool,
        use_gpu: bool,
        config: Config,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.config = config
        self.net = transformer
        self.io_binding = self.net.io_binding()
        self.max_seq_len = config.max_seq_len
        self.eos_token = config.eos_token

        self.inv_rhythm_vocab = {v: k for k, v in config.rhythm_vocab.items()}
        self.inv_pitch_vocab = {v: k for k, v in config.pitch_vocab.items()}
        self.inv_lift_vocab = {v: k for k, v in config.lift_vocab.items()}
        self.inv_articulation_vocab = {v: k for k, v in config.articulation_vocab.items()}
        self.inv_position_vocab = {v: k for k, v in config.position_vocab.items()}

        self.fp16 = fp16
        self.use_gpu = use_gpu
        self.device_id = 0
        self.output_names = [
            "out_rhythms",
            "out_pitchs",
            "out_lifts",
            "out_positions",
            "out_articulations",
        ]

    def generate(
        self,
        context: NDArray
    ) -> list[EncodedSymbol]:
        start_tokens = np.array([[1]], dtype=np.int64)
        nonote_tokens = np.array([[0]], dtype=np.int64)

        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        out_rhythm = start_tokens
        out_pitch = nonote_tokens
        out_lift = nonote_tokens
        out_articulations = nonote_tokens
        cache, kv_input_names, kv_output_names = self.init_cache()
        output_names = self.output_names + kv_output_names
        context_reduced = context[:, :1]

        symbols: list[EncodedSymbol] = []

        for step in range(self.max_seq_len):
            # after the first step we don't pass the full context into the decoder
            # x_transformers uses [:, :0] to split the context
            # which caused a Reshape error when loading the onnx model
            context = context if step == 0 else context_reduced

            # Bind Inputs
            self.io_binding.bind_cpu_input("rhythms", out_rhythm)
            self.io_binding.bind_cpu_input("pitchs", out_pitch)
            self.io_binding.bind_cpu_input("lifts", out_lift)
            self.io_binding.bind_cpu_input("articulations", out_articulations)
            self.io_binding.bind_cpu_input("context", context)
            self.io_binding.bind_cpu_input("cache_len", np.array([step], dtype=np.int64))
            for name, cache_val in zip(kv_input_names, cache, strict=True):
                self.io_binding.bind_ortvalue_input(name, cache_val)

            # Bind Outputs
            for name in output_names:
                self.io_binding.bind_output(name, "cuda" if self.use_gpu else "cpu", self.device_id)

            # Run inference
            self.net.run_with_iobinding(iobinding=self.io_binding)

            # Get outputs
            outputs = self.io_binding.get_outputs()
            cache = outputs[5:]

            # Greedy decoding: pick the highest logit directly for each output
            rhythmsp = outputs[0].numpy()
            pitchsp = outputs[1].numpy()
            liftsp = outputs[2].numpy()
            positionsp = outputs[3].numpy()
            articulationsp = outputs[4].numpy()
            attention = outputs[5].numpy()


            lift_token = detokenize_single(liftsp, self.inv_lift_vocab)
            pitch_token = detokenize_single(pitchsp, self.inv_pitch_vocab)
            rhythm_token = detokenize_single(rhythmsp, self.inv_rhythm_vocab)
            articulation_token = detokenize_single(articulationsp, self.inv_articulation_vocab)
            position_token = detokenize_single(positionsp, self.inv_position_vocab)

            if rhythmsp == self.eos_token:
                break

            symbol = EncodedSymbol(
                rhythm=rhythm_token,
                pitch=pitch_token,
                lift=lift_token,
                articulation=articulation_token,
                position=position_token,
                coordinates=attention,
            )
            symbols.append(symbol)

            out_lift = np.array([liftsp])
            out_pitch = np.array([pitchsp])
            out_rhythm = np.array([rhythmsp])
            out_articulations = np.array([articulationsp])

        return symbols

    def init_cache(self, cache_len: int = 0) -> tuple[list[NDArray], list[str], list[str]]:
        cache = []
        input_names = []
        output_names = []
        heads = self.config.decoder_heads
        head_dim = self.config.decoder_dim // heads
        for i in range(self.config.decoder_depth * 4):
            if self.fp16:  # the cache needs to be fp16 as well
                cache.append(
                    ort.OrtValue.ortvalue_from_numpy(
                        np.zeros((1, heads, cache_len, head_dim), dtype=np.float16),
                        "cuda" if self.use_gpu else "cpu",
                        self.device_id,
                    )
                )
            else:
                cache.append(
                    ort.OrtValue.ortvalue_from_numpy(
                        np.zeros((1, heads, cache_len, head_dim), dtype=np.float32),
                        "cuda" if self.use_gpu else "cpu",
                        self.device_id,
                    )
                )
            input_names.append(f"cache_in{i}")
            output_names.append(f"cache_out{i}")
        return cache, input_names, output_names


def detokenize_single(token: np.integer | int, vocab: dict[int, str]) -> str | None:
    tok = vocab[int(token)]
    if tok in ("[BOS]", "[EOS]", "[PAD]"):
        return None
    return tok


def get_decoder(config: Config) -> ScoreDecoder:
    """
    Returns Tromr's Decoder
    """
    use_gpu = False
    if config.use_gpu_inference:
        try:
            onnx_transformer = ort.InferenceSession(
                config.filepaths.decoder_path_fp16, providers=["CUDAExecutionProvider"]
            )
            fp16 = True
            # Sometimes Ort falls automatically back to the CPU EP
            # if so we get an error due to the device selection in init_cache()
            if "CUDAExecutionProvider" in onnx_transformer.get_providers():
                use_gpu = True
            else:
                eprint(
                    "Onnxruntime is not using GPU and therefore falling back to CPU. This is slow."
                )

        except Exception as ex:
            eprint(ex)
            eprint("Going on without GPU support")
            onnx_transformer = ort.InferenceSession(config.filepaths.decoder_path_fp16)
            fp16 = True

    else:
        onnx_transformer = ort.InferenceSession(config.filepaths.decoder_path)
        fp16 = False

    return ScoreDecoder(onnx_transformer, fp16, use_gpu, config=config)