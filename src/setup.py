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
    package_data={"stretch": ["config/**/*.yaml", "perception/*.tsv"]},
    install_requires=[
        # Machine learning code
        "torch<2.4",
        "torchvision",
        # General utilities
        "pyyaml",
        "pyzmq",
        "numpy<2",
        "numba",
        "opencv-python",
        "scipy",
        "matplotlib",
        "trimesh",
        "yacs",
        "scikit-image",
        "sophuspy",
        "pin",  # Pinocchio IK solver
        "pynput",
        "pyusb",
        "schema",
        "overrides",
        "wget",
        # From openai
        "openai",
        "openai-clip",
        # Hardware dependencies
        "hello-robot-stretch-urdf",
        "pyrealsense2",
        "urchin",
        # Visualization
        "rerun-sdk",
        # For siglip encoder
        "sentencepiece",
        # For git tools
        "gitpython",
        # Configuration tools and neural networks
        "hydra-core",
        "timm",
        "huggingface_hub[cli]",
        "transformers",
        "accelerate",
        "einops",
        # Compression tools
        "pyliblzfse",
        "webp",
        # UI tools
        "termcolor",
        # Audio
        "google-cloud-texttospeech",  # online TTS engine, requiring credentials.
        "gtts",  # online TTS engine, not requiring credentials.
        "librosa",  # audio analysis (e.g., spectral similarity)
        "PyAudio>=0.2.14",  # the version specification is necessary because apt has 0.2.12 which is incompatible with recent numpy
        "openai-whisper",
        "overrides",  # better inheritance of docstrings
        "pydub",  # playback audio
        "pyttsx3",  # offline TTS engine. TODO: There are better options, such as "tts_models/en/ljspeech/fast_pitch" from https://github.com/coqui-ai/TTS
        "simpleaudio",  # playback audio
        "sounddevice",  # Suppresses ALSA warnings when launching PyAudio
        "wave",
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
            "lark",
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
