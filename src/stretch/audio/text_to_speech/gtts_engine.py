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
from io import BytesIO
from typing import Any, Optional

# Third-party imports
import simpleaudio
import sounddevice  # suppress ALSA warnings # noqa: F401
from gtts import gTTS
from overrides import override
from pydub import AudioSegment

# Local imports
from ..base import AbstractTextToSpeech

# Create the default logger
logging.basicConfig(level=logging.INFO)
DEFAULT_LOGGER = logging.getLogger(__name__)


class GTTSTextToSpeech(AbstractTextToSpeech):
    """
    Text-to-speech engine using gTTS.
    """

    # Set a common framerate that's widely useful
    target_frame_rate: int = 44100

    @override  # inherit the docstring from the parent class
    def __init__(self, logger: logging.Logger = DEFAULT_LOGGER):
        super().__init__(logger)
        self._can_say_async = True

        # Initialize the voices.
        # https://gtts.readthedocs.io/en/latest/module.html#gtts.lang.tts_langs
        self._voice_ids = [
            "com",  # Default
            "us",  # United States
            "com.au",  # Australia
            "co.uk",  # United Kingdom
            "ca",  # Canada
            "co.in",  # India
            "ie",  # Ireland
            "co.za",  # South Africa
            "com.ng",  # Nigeria
        ]
        self.voice_id = "com"
        self._playback: Optional[simpleaudio.PlayObject] = None

    def __synthesize_text(self, text: str) -> gTTS:
        """
        Synthesize the given text.

        Parameters
        ----------
        text : str
            The text to speak.

        Returns
        -------
        gTTS
            The synthesized text.
        """
        return gTTS(text=text, lang="en", tld=self.voice_id, slow=self.is_slow)

    def __play_text(self, tts: gTTS) -> None:
        """
        Play the synthesized text.

        Parameters
        ----------
        tts : gTTS
            The synthesized text.
        """
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio = AudioSegment.from_file(fp, format="mp3")

        if audio.frame_rate != self.target_frame_rate:
            audio = audio.set_frame_rate(self.target_frame_rate)

        self._playback = simpleaudio.play_buffer(
            audio.raw_data, audio.channels, audio.sample_width, audio.frame_rate
        )

    @override  # inherit the docstring from the parent class
    def say_async(self, text: str) -> None:
        tts = self.__synthesize_text(text)
        self.__play_text(tts)

    @override  # inherit the docstring from the parent class
    def is_speaking(self) -> bool:
        if self._playback is None:
            return False
        if not self._playback.is_playing():
            self._playback = None
            return False
        return True

    @override  # inherit the docstring from the parent class
    def say(self, text: str) -> None:
        tts = self.__synthesize_text(text)
        self.__play_text(tts)
        self._playback.wait_done()
        self._playback = None

    @override  # inherit the docstring from the parent class
    def stop(self):
        if self._playback is not None:
            self._playback.stop()
            self._playback = None

    @override  # inherit the docstring from the parent class
    def save_to_file(self, text: str, filepath: str, **kwargs: Any) -> None:
        if not self.is_file_type_supported(filepath):
            return
        tts = self.__synthesize_text(text)
        tts.save(filepath)
