import os
from time import perf_counter

import numpy as np
from PIL import Image

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.transformer.decoder_inference import get_decoder
from homr.transformer.encoder_inference import Encoder
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray


class Staff2Score:
    """
    Inference class for Tromr. Use predict() for prediction
    """

    def __init__(self, use_gpu: bool = True) -> None:
        self.config = Config()
        self.encoder = Encoder()
        self.decoder = get_decoder(self.config)

        if not os.path.exists(self.config.filepaths.rhythmtokenizer):
            raise RuntimeError(
                "Failed to find tokenizer config" + self.config.filepaths.rhythmtokenizer
            )  # noqa: E501

    def predict(self, image: NDArray) -> list[EncodedSymbol]:
        """
        Inference an image (NDArray) using Tromr.
        """
        x = _transform(image=image).transpose(0, 3, 1, 2)

        t0 = perf_counter()

        # Create special tokens
        start_token = np.array([[1]], dtype=np.int64)
        nonote_token = np.array([[0]], dtype=np.int64)

        # Generate context with encoder
        context = self.encoder.generate(x)

        # Make a prediction using decoder
        t1 = perf_counter()
        out = self.decoder.generate(
            start_token,
            nonote_token,
            seq_len=self.config.max_seq_len,
            eos_token=self.config.eos_token,
            context=context,
        )

        return out, perf_counter() - t1


class ConvertToArray:
    def __init__(self) -> None:
        self.mean = np.array([0.7931]).reshape(1, 1, 1)
        self.std = np.array([0.1738]).reshape(1, 1, 1)

    def normalize(self, array: NDArray) -> NDArray:
        return (array - self.mean) / self.std

    def __call__(self, image: NDArray) -> NDArray:
        arr = np.array(image) / 255
        arr = arr[np.newaxis, :, :, np.newaxis]
        return self.normalize(arr).astype(np.float32)


_transform = ConvertToArray()


def test_transformer_on_image(path_to_img: str) -> None:
    """
    Tests the transformer on an image and prints the results.
    Args:
        path_to_img(str): Path to the image to test
    """
    from PIL import Image

    model = Staff2Score(False)
    image = Image.open(path_to_img)
    out = model.predict(np.array(image))
    eprint(out)


def benchmark(path_to_img: str, it) -> None:
    from PIL import Image

    model = Staff2Score(False)
    image = Image.open(path_to_img)
    t_complete = 0
    for i in range(it):
        t_complete += model.predict(np.array(image))[1]
    
    eprint(t_complete/it)


if __name__ == "__main__":
    benchmark("staff.png", 10)
