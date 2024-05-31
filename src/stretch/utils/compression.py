import io
import pickle

import numpy as np
import webp
from PIL import Image


## Compress Python Object to Bytes
def zip(obj):
    """
    Compresses a Python object to bytes using pickle.

    Args:
        obj: The Python object to be compressed.

    Returns:
        bytes: The compressed bytes representation of the object.
    """
    compressed_bytes = pickle.dumps(obj)
    return compressed_bytes


## Decompress Bytes to Python Object
def unzip(compressed_bytes):
    """
    Decompresses bytes to a Python object using pickle.

    Args:
        compressed_bytes: The compressed bytes representation of the object.

    Returns:
        The decompressed Python object.
    """
    obj = pickle.loads(compressed_bytes)
    return obj


def to_webp(numpy_array):
    pil_img = Image.fromarray(numpy_array)
    pic = webp.WebPPicture.from_pil(pil_img)
    config = webp.WebPConfig.new(quality=80)
    webp_data = pic.encode(config).buffer()
    return webp_data


def from_webp(webp_data) -> np.ndarray:
    webp_file = io.BytesIO(webp_data)
    webp_data = webp.WebPData.from_buffer(webp_file.read())
    numpy_array = webp_data.decode_ndarray()
    return numpy_array
