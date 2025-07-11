# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import setuptools

__version__ = None
with open("stretch/version.py") as f:
    exec(f.read())  # overrides __version__

with open("../README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stretch_ai",
    version=__version__,
    author="Hello Robot Inc.",
    author_email="support@hello-robot.com",
    description="Stretch Python API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hello-robot/stretchpy",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={"stretch": ["config/**/*.yaml", "perception/*.tsv", "simulation/models/*"]},
    install_requires=[
        # Machine learning code, we will install these packages in install.sh instead
        "torch>=2.6",
        "torchvision",
        "torchaudio",
        # General utilities
        "pyyaml",
        "pyzmq",
        "numpy<2",
        "numba",
        "opencv-python",
        "scipy",
        "matplotlib",
        "trimesh>=3.10.0",
        "yacs",
        "scikit-image>=0.21.0",
        "sophuspy",
        "pin",  # Pinocchio IK solver
        "pynput",
        "pyusb",
        "schema",
        "overrides",
        "wget",
        # From openai
        "openai >= 1.88.0",
        "openai-clip",
        # For gemini
        "google-genai",
        # For Yolo
        "ultralytics==8.3.161",
        # Hardware dependencies
        "hello-robot-stretch-urdf",
        "pyrealsense2",
        "urchin",
        # Visualization
        "rerun-sdk==0.18.0",
        # For siglip encoder
        "sentencepiece",
        # For git tools
        "gitpython",
        # Configuration tools and neural networks
        "hydra-core",
        "timm>1.0.0",
        "huggingface_hub[cli]>=0.24.7",
        "open-clip-torch>=2.32.0",
        "transformers>=4.50.0",
        "retry",
        "qwen_vl_utils",
        "bitsandbytes",
        "triton >= 2.3.1",
        "accelerate >= 1.5.0",
        "einops",
        "protobuf",
        # Meta neural nets
        "segment-anything",
        # Compression tools
        "pyliblzfse",
        "webp>=0.3.0",
        # UI tools
        "termcolor",
        # Audio
        "librosa",  # audio analysis (e.g., spectral similarity)
        "PyAudio>=0.2.14",  # the version specification is necessary because apt has 0.2.12 which is incompatible with recent numpy
        "openai-whisper",
        "overrides",  # better inheritance of docstrings
        "pydub",  # playback audio
        "simpleaudio",  # playback audio
        # "wave",
        # These are not supported > python 3.11
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
            "lark",
        ],
        "discord": [
            "discord.py",
            "python-dotenv",
        ],
        "sim": [
            "mujoco",
            "hello-robot-stretch-urdf",
            "grpcio",
        ],
        "hand_tracker": [
            "mediapipe",
            "webcam",
        ],
    },
)
