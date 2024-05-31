import pickle


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
