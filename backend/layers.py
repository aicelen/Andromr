from typing import List

import numpy as np
from numpy import ndarray


# Dictionary of numpy.ndarray, representing all kinds of information
# extracted by different extractors.
_layers = {}

_access_count = {}


def register_layer(name: str, layer: ndarray) -> None:
    if name in _layers:
        print("Name already registered! Choose another name or delete it first.")
        return

    assert isinstance(layer, np.ndarray)
    _layers[name] = layer
    _access_count[name] = 0


def get_layer(name: str) -> ndarray:
    if name not in _layers:
        raise KeyError(f"The given layer name not registered: {name}")
    _access_count[name] += 1
    return _layers[name]


def delete_layer(name):
    if name in _layers:
        del _layers[name]
        del _access_count[name]


def list_layers() -> List:
    return list(_layers.keys())


def show_access_count():
    print(_access_count)
