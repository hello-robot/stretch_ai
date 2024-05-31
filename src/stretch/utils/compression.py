import io
import pickle

import liblzfse
import numpy as np
import webp
from PIL import Image


## Compress Python Object to Bytes
def zip_depth(obj: np.ndarray):
    """
    Compresses a Python object to bytes using pickle.

    Args:
        obj: The Python object to be compressed.

    Returns:
        bytes: The compressed bytes representation of the object.
    """
    # compressed_bytes = pickle.dumps(obj)
    compressed_bytes = liblzfse.compress(obj.astype(np.uint16).tobytes())
    # depth_bytes = liblzfse.compress(depth_array.astype(np.float32).tobytes())
    return compressed_bytes


## Decompress Bytes to Python Object
def unzip_depth(compressed_bytes):
    """
    Decompresses bytes to a Python object using pickle.

    Args:
        compressed_bytes: The compressed bytes representation of the object.

    Returns:
        The decompressed Python object.
    """
    # obj = pickle.loads(compressed_bytes)
    buffer = np.frombuffer(liblzfse.decompress(compressed_bytes), dtype=np.uint16)
    return buffer


def to_webp(img: np.ndarray):
    """
    Converts a NumPy array to a WebP image (bytes).

    Args:
        arr (numpy.ndarray): The input NumPy array.

    Returns:
        bytes: The WebP image data as bytes.
    """
    # Convert the NumPy array to a PIL Image
    img = Image.fromarray(img)

    # Create a BytesIO object to store the WebP image data
    webp_bytes = io.BytesIO()

    # Save the image as WebP format to the BytesIO object
    img.save(webp_bytes, format="WebP", lossless=True)

    # Get the bytes from the BytesIO object
    webp_bytes = webp_bytes.getvalue()
    return webp_bytes


def from_webp(webp_data) -> np.ndarray:
    # Create a BytesIO object from the WebP image data
    webp_io = io.BytesIO(webp_data)

    # Open the WebP image from the BytesIO object
    img = Image.open(webp_io)

    # Convert the PIL Image to a NumPy array
    arr = np.array(img)
    return arr
