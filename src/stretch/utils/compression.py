# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import io
from typing import Optional, Tuple, Union

import cv2
import liblzfse
import numpy as np
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
def unzip_depth(compressed_bytes, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Decompresses bytes to a Python object using pickle.

    Args:
        compressed_bytes: The compressed bytes representation of the object.

    Returns:
        The decompressed Python object.
    """
    # obj = pickle.loads(compressed_bytes)
    buffer = np.frombuffer(liblzfse.decompress(compressed_bytes), dtype=np.uint16)
    if shape is not None:
        buffer = buffer.reshape(*shape)
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
    pil_img = Image.fromarray(img)

    # Create a BytesIO object to store the WebP image data
    webp_bytes = io.BytesIO()

    # Save the image as WebP format to the BytesIO object
    pil_img.save(webp_bytes, format="WebP", lossless=False)

    # Get the bytes from the BytesIO object
    webp_bytes_data = webp_bytes.getvalue()
    return webp_bytes_data


def from_webp(webp_data) -> np.ndarray:
    # Create a BytesIO object from the WebP image data
    webp_io = io.BytesIO(webp_data)

    # Open the WebP image from the BytesIO object
    img = Image.open(webp_io)

    # Convert the PIL Image to a NumPy array
    arr = np.array(img)
    return arr


def to_jp2(image: np.ndarray, quality: int = 800):
    """Depth is better encoded as jp2"""
    _, compressed_image = cv2.imencode(
        ".jp2", image, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, quality]
    )
    return compressed_image


def to_jpg(image: np.ndarray, quality: int = 90):
    """Encode as jpeg"""
    _, compressed_image = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return compressed_image


def from_jpg(compressed_image: Union[bytes, np.ndarray]) -> np.ndarray:
    """Convert compressed image to numpy array"""
    if isinstance(compressed_image, bytes):
        compressed_image = np.frombuffer(compressed_image, dtype=np.uint8)
    return cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)


def from_jp2(compressed_image: Union[bytes, np.ndarray]) -> np.ndarray:
    """Convert compressed image to numpy array"""
    if isinstance(compressed_image, bytes):
        compressed_image = np.frombuffer(compressed_image, dtype=np.uint8)
    return cv2.imdecode(compressed_image, cv2.IMREAD_UNCHANGED)
