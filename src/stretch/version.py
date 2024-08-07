# Store the version here so:
# 1) we don't load dependencies by storing it in __init__.py
# 2) we can import it in setup.py for the same reason
# 3) we can import it into your module

__version__ = "0.0.8"
__stretchpy_protocol__ = "spp0"

if __name__ == "__main__":
    print(__version__)
