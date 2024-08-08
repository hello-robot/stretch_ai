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
import traceback
from io import BytesIO
from typing import Any, Optional

# Third-party imports
import simpleaudio
import sounddevice  # suppress ALSA warnings # noqa: F401
from google.cloud import texttospeech
from overrides import override
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# Local imports
from ..base import AbstractTextToSpeech

# Create the default logger
logging.basicConfig(level=logging.INFO)
DEFAULT_LOGGER = logging.getLogger(__name__)


class GoogleCloudTextToSpeech(AbstractTextToSpeech):
    """
    Text-to-speech engine using Google Cloud.
    """

    @override  # inherit the docstring from the parent class
    def __init__(self, logger: logging.Logger = DEFAULT_LOGGER):
        super().__init__(logger)
        self._can_say_async = True

        # Initialize the voices and speeds.
        # https://cloud.google.com/text-to-speech/docs/voices
        # fmt: off
        self._voice_ids = [
            "en-US-Casual-K",   "en-US-Journey-D",  "en-US-Journey-F",
            "en-US-Journey-O",  "en-US-Neural2-A",  "en-US-Neural2-C",
            "en-US-Neural2-D",  "en-US-Neural2-E",  "en-US-Neural2-F",
            "en-US-Neural2-G",  "en-US-Neural2-H",  "en-US-Neural2-I",
            "en-US-Neural2-J",  "en-US-News-K",     "en-US-News-L",
            "en-US-News-N",     "en-US-Polyglot-1", "en-US-Standard-A",
            "en-US-Standard-B", "en-US-Standard-C", "en-US-Standard-D",
            "en-US-Standard-E", "en-US-Standard-F", "en-US-Standard-G",
            "en-US-Standard-H", "en-US-Standard-I", "en-US-Standard-J",
            "en-US-Studio-O",   "en-US-Studio-Q",   "en-US-Wavenet-A",
            "en-US-Wavenet-B",  "en-US-Wavenet-C",  "en-US-Wavenet-D",
            "en-US-Wavenet-E",  "en-US-Wavenet-F",  "en-US-Wavenet-G",
            "en-US-Wavenet-H",  "en-US-Wavenet-I",  "en-US-Wavenet-J"
        ]
        # fmt: on
        self.voice_id = "en-US-Neural2-I"
        self.is_slow = False
        self._playback: Optional[simpleaudio.PlayObject] = None

        # Create a client and its configurations
        self._client = texttospeech.TextToSpeechClient()

    @AbstractTextToSpeech.voice_id.setter  # type: ignore
    @override  # inherit the docstring from the parent class
    def voice_id(self, voice_id: str) -> None:
        AbstractTextToSpeech.voice_id.fset(self, voice_id)
        self._voice_config = GoogleCloudTextToSpeech._get_voice_config(self.voice_id)

    @staticmethod
    def _get_voice_config(voice_id: str) -> texttospeech.VoiceSelectionParams:
        """
        Get the voice configuration for the given voice ID.

        Parameters
        ----------
        voice_id : str
            The voice ID.

        Returns
        -------
        texttospeech.VoiceSelectionParams
            The voice configuration.
        """
        voice_components = voice_id.split("-")
        langauge_code = "-".join(voice_components[0:2])
        return texttospeech.VoiceSelectionParams(
            language_code=langauge_code,
            name=voice_id,
        )

    @AbstractTextToSpeech.is_slow.setter  # type: ignore
    @override  # inherit the docstring from the parent class
    def is_slow(self, is_slow: bool) -> None:
        AbstractTextToSpeech.is_slow.fset(self, is_slow)
        self._audio_config = GoogleCloudTextToSpeech._get_audio_config(self.is_slow)

    @staticmethod
    def _get_audio_config(is_slow: bool) -> texttospeech.AudioConfig:
        """
        Get the audio configuration.

        Parameters
        ----------
        is_slow : bool
            Whether or not the speech is slow.

        Returns
        -------
        texttospeech.AudioConfig
            The audio configuration.
        """
        return texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.7 if is_slow else 1.0,
        )

    def __synthesize_text(self, text: str) -> bytes:
        """
        Synthesize the given text.

        Parameters
        ----------
        text : str
            The text to synthesize.

        Returns
        -------
        bytes
            The audio content.
        """
        synthesis_input = texttospeech.SynthesisInput(text=text)
        response = self._client.synthesize_speech(
            input=synthesis_input, voice=self._voice_config, audio_config=self._audio_config
        )
        return response.audio_content

    def __play_text(self, audio_bytes: bytes) -> None:
        """
        Play the given audio bytes.

        Parameters
        ----------
        audio_bytes : bytes
            The audio bytes.
        """
        fp = BytesIO()
        fp.write(audio_bytes)
        fp.flush()
        fp.seek(0)
        try:
            audio = AudioSegment.from_file(fp, format="mp3")
        except CouldntDecodeError as err:
            self._logger.error(traceback.format_exc())
            self._playback = None
            return
        self._playback = simpleaudio.play_buffer(
            audio.raw_data, audio.channels, audio.sample_width, audio.frame_rate
        )

    @override  # inherit the docstring from the parent class
    def say_async(self, text: str) -> None:
        audio_bytes = self.__synthesize_text(text)
        self.__play_text(audio_bytes)

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
        audio_bytes = self.__synthesize_text(text)
        self.__play_text(audio_bytes)
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
        audio_bytes = self.__synthesize_text(text)
        with open(filepath, "wb") as f:
            f.write(audio_bytes)
