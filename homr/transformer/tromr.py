import numpy as np

from globals import appdata
from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray
from homr.simple_logging import eprint

from kivy.utils import platform

if platform == "android":
    from jnius import autoclass  # type: ignore

    Staff2Score = autoclass("com.aicelen.andromr.Staff2Score")
    ByteBuffer = autoclass("java.nio.ByteBuffer")
    ByteOrder = autoclass("java.nio.ByteOrder")
else:
    from homr.transformer.decoder_inference import get_decoder
    from homr.transformer.encoder_inference import Encoder


model = None


class TromrAndroid:
    """
    Wrapper around Staff2Score.java
    How it works:
    1. Run encoder (java)
    2. Run decoder (still java)
    3. Detokenize the result (python)
    """

    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        self.config = config
        self.num_threads = appdata.threads

        self.inv_rhythm_vocab = {v: k for k, v in config.rhythm_vocab.items()}
        self.inv_pitch_vocab = {v: k for k, v in config.pitch_vocab.items()}
        self.inv_lift_vocab = {v: k for k, v in config.lift_vocab.items()}
        self.inv_articulation_vocab = {v: k for k, v in config.articulation_vocab.items()}
        self.inv_position_vocab = {v: k for k, v in config.position_vocab.items()}
        self.inv_slur_vocab = {v: k for k, v in config.slur_vocab.items()}

        self.model = Staff2Score()
        self.model.load(
            str(config.filepaths.encoder_path),
            str(config.filepaths.decoder_path),
            str(self.num_threads),
        )

    def run(self, image: NDArray):
        image = np.ascontiguousarray(image)
        if image.dtype == np.float32:
            flat = image.ravel()
            buffer_bytes = flat.tobytes()
            java_byte_buffer = ByteBuffer.wrap(buffer_bytes)
            java_byte_buffer.order(ByteOrder.nativeOrder())
            float_buffer = java_byte_buffer.asFloatBuffer()

        else:
            raise TypeError(f"Expected np.float32 array but got {image.dtype}")

        result = self.model.run(float_buffer)
        tokens = np.array(result, dtype=np.int64).reshape([6, self.config.max_seq_len])

        symbols: list[EncodedSymbol] = []
        for i in range(self.config.max_seq_len):
            rhythm = tokens[0][i] - 1
            lift = tokens[1][i] - 1
            pitch = tokens[2][i] - 1
            articulation = tokens[3][i] - 1
            pos = tokens[4][i] - 1
            slur = tokens[5][i] - 1

            # Java Array was filled with 0, we did +1 to every generated
            # token (starting from 0). Therefore -1 indicates that this
            # is part of the array prefill and the generated tokens stop there
            if rhythm in (-1, self.config.eos_token):
                break

            rhythm_token = detokenize_single(rhythm, self.inv_rhythm_vocab)
            lift_token = detokenize_single(lift, self.inv_lift_vocab)
            pitch_token = detokenize_single(pitch, self.inv_pitch_vocab)
            articulation_token = detokenize_single(articulation, self.inv_articulation_vocab)
            position_token = detokenize_single(pos, self.inv_position_vocab)
            slur_token = detokenize_single(slur, self.inv_slur_vocab)

            symbol = EncodedSymbol(
                rhythm=rhythm_token,
                pitch=pitch_token,
                lift=lift_token,
                articulation=articulation_token,
                position=position_token,
                slur=slur_token,
                coordinates=None,
            )
            symbols.append(symbol)

        return symbols

    def unload_model(self):
        self.model.unload_model()


class TromrDesktop:
    def __init__(self, config: Config):
        self.encoder = Encoder()
        self.decoder = get_decoder(config)
        self.num_threads = appdata.threads

    def run(self, image: NDArray):
        ctx = self.encoder.generate(image)
        return self.decoder.generate(ctx)


def detokenize_single(token: np.integer | int, vocab: dict[int, str]) -> str | None:
    tok = vocab[int(token)]
    if tok in ("[BOS]", "[EOS]", "[PAD]"):
        return None
    return tok


def get_tromr(config: Config) -> TromrAndroid | TromrDesktop:
    """
    Get Tromr model returns the correct model depending on the platform
    """
    global model
    if model is None or model.num_threads != appdata.threads:
        if platform == "android":
            if model is not None:
                model.unload_model()
                model = None
            model = TromrAndroid(config=config)
        else:
            model = TromrDesktop(config=config)
        eprint(f"Loaded Tromr runnning on {model.num_threads} threads")
    return model
