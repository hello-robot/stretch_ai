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
import time
from typing import Any

# Third-party imports
import pyttsx3
import sounddevice  # suppress ALSA warnings # noqa: F401
from overrides import override

# Local imports
from ..base import AbstractTextToSpeech

# Create the default logger
logging.basicConfig(level=logging.INFO)
DEFAULT_LOGGER = logging.getLogger(__name__)


class PyTTSx3TextToSpeech(AbstractTextToSpeech):
    """
    Text-to-speech engine using pyttsx3. A big benefit of pyttsx3 compared
    to other enginers is that it runs offline. However, its Linux voices tend
    to be less natural than other engines.
    """

    @override  # inherit the docstring from the parent class
    def __init__(self, logger: logging.Logger = DEFAULT_LOGGER):
        super().__init__(logger)
        self._engine = pyttsx3.init()

        # Initialize the voices
        voices = self._engine.getProperty("voices")
        # Variants documentation: https://espeak.sourceforge.net/languages.html
        variants = [
            "m1",
            "m2",
            "m3",
            "m4",
            "m5",
            "m6",
            "m7",
            "f1",
            "f2",
            "f3",
            "f4",
            "croak",
            "whisper",
        ]
        for voice in voices:
            self._voice_ids.append(voice.id)
            for variant in variants:
                self._voice_ids.append(voice.id + "+" + variant)
        self.voice_id = "default"

        # Initialize the speeds
        self.slow_speed = 100  # wpm
        self.default_speed = 150  # wpm

    @AbstractTextToSpeech.voice_id.setter  # type: ignore
    @override  # inherit the docstring from the parent class
    def voice_id(self, voice_id: str) -> None:
        AbstractTextToSpeech.voice_id.fset(self, voice_id)
        self._engine.setProperty("voice", voice_id)

    @AbstractTextToSpeech.is_slow.setter  # type: ignore
    @override  # inherit the docstring from the parent class
    def is_slow(self, is_slow: bool) -> None:
        AbstractTextToSpeech.is_slow.fset(self, is_slow)
        if is_slow:
            self._engine.setProperty("rate", self.slow_speed)
        else:
            self._engine.setProperty("rate", self.default_speed)

    @override  # inherit the docstring from the parent class
    def say_async(self, text: str) -> None:
        self._logger.warning("Asynchronous speaking is not supported for PyTTSx3 on Linux.")

    @override  # inherit the docstring from the parent class
    def is_speaking(self) -> bool:
        # Because asynchronous speaking is not supported in pyttsxy on Linux,
        # if this function is called, it is assumed that the engine is not speaking.
        # This works as long as `is_speaking` and `say` will be called from
        # the same thread.
        return False

    @override  # inherit the docstring from the parent class
    def say(self, text: str) -> None:
        self._engine.say(text)
        self._engine.runAndWait()

    @override  # inherit the docstring from the parent class
    def stop(self) -> None:
        # Although interruptions are nominally supported in pyttsx3
        # (https://pyttsx3.readthedocs.io/en/latest/engine.html#examples),
        # in practice, the Linux implementation spins of an ffmpeg process
        # which can't be interrupted in its current implementation:
        # https://github.com/nateshmbhat/pyttsx3/blob/5d3755b060a980f48fcaf81df018dd06cbd17a8f/pyttsx3/drivers/espeak.py#L175 # noqa: E501
        self._logger.warning("Asynchronous stopping is not supported for PyTTSx3 on Linux.")

    @override  # inherit the docstring from the parent class
    def save_to_file(self, text: str, filepath: str, **kwargs: Any) -> None:
        if not self.is_file_type_supported(filepath):
            return  # error message already logged

        # Get the parameters. In practice, because pyttsx3 spawns an ffmpeg
        # process when saving, it does not wait for the process to finish.
        # Therefore, the sleep_secs parameter is used to wait for some time
        # before returning.
        sleep_secs = 2.0
        if "sleep_secs" in kwargs:
            sleep_secs = float(kwargs["sleep_secs"])

        self._engine.proxy.setBusy(True)
        self._engine.save_to_file(text, filepath)
        self._engine.runAndWait()
        time.sleep(sleep_secs)
        self._engine.proxy.setBusy(False)
