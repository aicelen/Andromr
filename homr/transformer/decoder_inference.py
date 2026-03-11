from typing import Any

import numpy as np
from homr.inference_engine import OnnxModel
from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray

from globals import appdata
from kivy import platform

decoder: OnnxModel | None = None


class ScoreDecoder:
    def __init__(
        self,
        transformer: OnnxModel,
        fp16: bool,
        use_gpu: bool,
        config: Config,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.config = config
        self.net = transformer
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
        start_tokens: NDArray,
        nonote_tokens: NDArray,
        **kwargs: Any,
    ) -> list[EncodedSymbol]:
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        out_rhythm = start_tokens
        out_pitch = nonote_tokens
        out_lift = nonote_tokens
        out_articulations = nonote_tokens
        cache, kv_input_names, kv_outputs_dicts = self.init_cache()
        context = kwargs["context"]
        context_reduced = kwargs["context"][:, :1]
        outputs_dict = {
            "out_rhythms": [1, 1, 260],
            "out_pitchs": [1, 1, 72],
            "out_lifts": [1, 1, 7],
            "out_positions": [1, 1, 3],
            "out_articulations": [1, 1, 144],
        }

        symbols: list[EncodedSymbol] = []

        for step in range(self.max_seq_len):
            x_lift = out_lift[:, -1:]  # for all: shape=(1,1)
            x_pitch = out_pitch[:, -1:]
            x_rhythm = out_rhythm[:, -1:]
            x_articulations = out_articulations[:, -1:]

            # after the first step we don't pass the full context into the decoder
            # x_transformers uses [:, :0] to split the context
            # which caused a Reshape error when loading the onnx model
            context = context if step == 0 else context_reduced

            inputs = {
                "rhythms": x_rhythm,
                "pitchs": x_pitch,
                "lifts": x_lift,
                "articulations": x_articulations,
                "context": context,
                "cache_len": np.array([step]),
            }

            if step == 0:
                # only in the first step the cache get's passed in (on android)
                for i in range(32):
                    inputs[kv_input_names[i]] = cache[i]

            else:
                # after the first step we don't pass the full context into the decoder
                # x_transformers uses [:, :0] to split the context
                # which caused a Reshape error when loading the onnx model
                context = context_reduced
                if platform != "android":
                    for i in range(32):
                        inputs[kv_input_names[i]] = cache[i]

            rhythmsp, pitchsp, liftsp, positionsp, articulationsp, *cache = self.net.run(
                inputs=inputs,
                outputs=outputs_dict | kv_outputs_dicts,  # merges the dicts
            )

            rhythm_sample = np.array([[rhythmsp[:, -1, :].argmax()]])
            pitch_sample = np.array([[pitchsp[:, -1, :].argmax()]])
            lift_sample = np.array([[liftsp[:, -1, :].argmax()]])
            articulation_sample = np.array([[articulationsp[:, -1, :].argmax()]])
            position_sample = np.array([[positionsp[:, -1, :].argmax()]])

            lift_token = detokenize(lift_sample, self.inv_lift_vocab)
            pitch_token = detokenize(pitch_sample, self.inv_pitch_vocab)
            rhythm_token = detokenize(rhythm_sample, self.inv_rhythm_vocab)
            articulation_token = detokenize(articulation_sample, self.inv_articulation_vocab)
            position_token = detokenize(position_sample, self.inv_position_vocab)

            if rhythm_sample[0][0] == self.eos_token:
                break

            symbol = EncodedSymbol(
                rhythm=rhythm_token[0],
                pitch=pitch_token[0],
                lift=lift_token[0],
                articulation=articulation_token[0],
                position=position_token[0],
                coordinates=None,
            )
            symbols.append(symbol)

            out_lift = np.concatenate((out_lift, lift_sample), axis=-1)
            out_pitch = np.concatenate((out_pitch, pitch_sample), axis=-1)
            out_rhythm = np.concatenate((out_rhythm, rhythm_sample), axis=-1)
            out_articulations = np.concatenate((out_articulations, articulation_sample), axis=-1)

        return symbols

    def init_cache(self, cache_len: int = 0) -> tuple[list[NDArray], list[str], list[str]]:
        cache = []
        input_names = []
        output_dict = {}
        for i in range(32):
            cache.append(np.zeros((1, 8, cache_len, 64), dtype=np.float32))
            input_names.append(f"cache_in{i}")
            output_dict[f"cache_out{i}"] = (
                f"cache_in{i}"  # this is used to check if this gets converted to python or not
            )
        return cache, input_names, output_dict


def detokenize(tokens: NDArray, vocab: dict[int, str]) -> list[str]:
    toks = [vocab[tok.item()] for tok in tokens]
    toks = [t for t in toks if t not in ("[BOS]", "[EOS]", "[PAD]")]
    return toks


def get_decoder(config: Config) -> ScoreDecoder:
    """
    Returns Tromr's Decoder
    """
    global decoder
    if decoder is None or decoder.num_threads != appdata.threads:
        decoder = OnnxModel(Config().filepaths.decoder_path)
        decoder.load()

    return ScoreDecoder(decoder, config=config, fp16=False, use_gpu=False)
