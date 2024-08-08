# Copyright 2024 Hello Robot Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.

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
        # Configuration tools and neural networks
        "hydra-core",
        "timm",
        "huggingface_hub[cli]",
        "transformers",
        "accelerate",
        # Compression tools
        "pyliblzfse",
        "webp",
        # UI tools
        "termcolor",
        # Audio
        "google-cloud-texttospeech",  # online TTS engine, requiring credentials.
        "gtts",  # online TTS engine, not requiring credentials.
        "librosa",  # audio analysis (e.g., spectral similarity)
        "PyAudio==0.2.14",  # the version specification is necessary because apt has 0.2.12 which is incompatible with recent numpy
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
        ]
    },
)
