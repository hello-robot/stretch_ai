# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Standard imports
import logging
import os
import stat
import subprocess
import tarfile
from typing import Any

import simpleaudio as sa
import wget
from overrides import override

# Local imports
from .base import AbstractTextToSpeech

# Create the default logger
logging.basicConfig(level=logging.INFO)
DEFAULT_LOGGER = logging.getLogger(__name__)


class PiperTextToSpeech(AbstractTextToSpeech):
    """
    Text-to-speech engine using gTTS.
    """

    @override  # inherit the docstring from the parent class
    def __init__(self, logger: logging.Logger = DEFAULT_LOGGER):
        super().__init__(logger)

        # Base directory for everything
        base_dir = "piper_tts"
        os.makedirs(base_dir, exist_ok=True)

        # 1) Download piper archive
        archive_name = "piper_amd64.tar.gz"
        archive_path = os.path.join(base_dir, archive_name)
        archive_url = "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz"

        if not os.path.exists(archive_path):
            print(f"Downloading {archive_name}...")
            wget.download(archive_url, out=archive_path)
            print("\nDownload complete.")
        else:
            print(f"{archive_name} already exists; skipping download.")

        # 2) Extract if needed
        piper_dir = os.path.join(base_dir, "piper")
        if not os.path.isdir(piper_dir):
            print(f"Extracting {archive_name} into {base_dir}/ …")
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(path=base_dir)
            print("Extraction complete.")
        else:
            print("piper directory already exists; skipping extraction.")

        # 3) Make piper executable
        piper_bin = os.path.join(piper_dir, "piper")
        if os.path.exists(piper_bin):
            # add owner‑execute bit (0o100) to whatever perms it already has
            st = os.stat(piper_bin).st_mode
            os.chmod(piper_bin, st | stat.S_IXUSR)
            print(f"Set executable bit on {piper_bin}.")
        else:
            print(f"WARNING: {piper_bin} not found!")

        self._piper_bin = piper_bin

        # 4) Download voice model + JSON
        # We currently only support the amy voice.
        files = {
            "en_US-amy-medium.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx?download=true",
            "en_US-amy-medium.onnx.json": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json?download=true",
        }

        for fname, url in files.items():
            dest = os.path.join(base_dir, fname)
            if not os.path.exists(dest):
                print(f"Downloading {fname}...")
                wget.download(url, out=dest)
            else:
                print(f"{fname} already exists; skipping download.")

        self._model_path = os.path.join(base_dir, "en_US-amy-medium.onnx")
        self.play_obj = None

        # Set a common framerate that's widely useful
        self.slow_speed = 16000
        self.default_speed = 22050
        self.sample_rate: int = self.default_speed

    @AbstractTextToSpeech.voice_id.setter  # type: ignore
    @override  # inherit the docstring from the parent class
    def voice_id(self, voice_id: str) -> None:
        self.logger.warning("Piper only supports one voice id.")

    @AbstractTextToSpeech.is_slow.setter  # type: ignore
    @override  # inherit the docstring from the parent class
    def is_slow(self, is_slow: bool) -> None:
        AbstractTextToSpeech.is_slow.fset(self, is_slow)
        if is_slow:
            self.sample_rate = self.slow_speed
        else:
            self.sample_rate = self.default_speed

    @override  # inherit the docstring from the parent class
    def say_async(self, text: str) -> None:
        wave_data = self.__generate_audio(text)
        self.__play_text(wave_data)

    def __generate_audio(self, text: str) -> bytes:
        proc = subprocess.run(
            [self._piper_bin, "--model", self._model_path, "--output-raw"],
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return proc.stdout

    @override  # inherit the docstring from the parent class
    def is_speaking(self) -> bool:
        if self.play_obj is None:
            return False
        if not self.play_obj.is_playing():
            self.play_obj = None
            return False
        return True

    def __play_text(self, raw_pcm: bytes) -> None:
        """
        Play the given audio bytes.
        """
        self.play_obj = sa.play_buffer(raw_pcm, 1, 2, self.sample_rate)

    @override  # inherit the docstring from the parent class
    def say(self, text: str) -> None:
        wave_data = self.__generate_audio(text)
        self.__play_text(wave_data)
        self.play_obj.wait_done()  # Wait until playback is finished
        self.play_obj = None

    @override  # inherit the docstring from the parent class
    def stop(self):
        if self.play_obj is not None:
            self.play_obj.stop()
            self.play_obj = None

    @override  # inherit the docstring from the parent class
    def save_to_file(self, text: str, filepath: str, **kwargs: Any) -> None:
        subprocess.call(
            f'echo "{text}" | '
            + self._piper_bin
            + " --model "
            + self._model_path
            + " --output_file "
            + filepath,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
