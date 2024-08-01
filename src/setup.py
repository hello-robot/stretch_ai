import setuptools

__version__ = None
with open("stretch/versions.py") as f:
    exec(f.read())  # overrides __version__

with open("../README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stretchpy",
    version=__version__,
    author="Hello Robot Inc.",
    author_email="support@hello-robot.com",
    description="Stretch Python API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hello-robot/stretchpy",
    packages=setuptools.find_packages(),
    install_requires=[
        # Machine learning code
        "torch",
        "torchvision",
        "pyyaml",
        "pyzmq",
        "numpy<2",
        "numba",
        "opencv-python",
        "scipy",
        "matplotlib",
        "trimesh",
        "openai-clip",
        "yacs",
        "loguru",
        "transformers",
        "scikit-image",
        "pybullet",
        "sophuspy",
        "pin",  # Pinocchio IK solver
        "pynput",
        "pyrealsense2",
        "urchin",
        "pyusb",
        "schema",
        # Visualization
        "rerun-sdk",
        # For siglip encoder
        "sentencepiece",
        # For git tools
        "gitpython",
        # Speech to text
        "openai-whisper",
        # Configuration tools and neural networks
        "hydra-core",
        "timm",
        "huggingface_hub[cli]",
        # Compression tools
        "pyliblzfse",
        "webp",
        # UI tools
        "termcolor",
        # Audio
        "pyaudio",
        "wave",
        "openai-whisper",
        # These are not supported in python 3.12
        "scikit-fmm",
        "open3d",
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "pytest",
            "flake8",
            "black",
            "mypy",
        ]
    },
)
