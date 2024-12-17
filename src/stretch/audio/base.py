# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Standard imports
import abc
import logging
from typing import Any, List

# Third-party imports
import numpy as np

# Create the default logger
logging.basicConfig(level=logging.INFO)
DEFAULT_LOGGER = logging.getLogger(__name__)


class AbstractSpeechToText(abc.ABC):
    """Basic speech to text module"""

    @abc.abstractmethod
    def process_audio(self, audio_data: np.ndarray) -> str:
        """Process audio data
        Args:
            audio_data: Audio data as a numpy array
        Returns
            str: Transcribed text
        """

    @abc.abstractmethod
    def transcribe_file(self, audio_file: str) -> str:
        """Transcribe the audio file to text.

        Args:
            audio_file (str): Path to the audio file.

        Returns
            str: Transcribed text.
        """


class AbstractTextToSpeech(abc.ABC):
    """
    Abstract base class for a text-to-speech engine that supports:
      - Setting the voice ID.
      - Setting the speed to default or slow.
      - Asynchronously speaking text.
      - Interrupting speech.
    """

    def __init__(self, logger: logging.Logger = DEFAULT_LOGGER):
        """
        Initialize the text-to-speech engine.

        Parameters
        ----------
        logger : logging.Logger
            The logger to use for logging messages.
        """
        self._logger = logger
        self._voice_ids: List[str] = []
        self._voice_id = ""
        self._is_slow = False

        # Whether or not this engine can speak asynchronously or not.
        self._can_say_async = False

    @property
    def voice_ids(self) -> List[str]:
        """
        Get the list of voice IDs available for the text-to-speech engine.

        Returns
        -------
        list[str]
            The list of voice IDs.
        """
        return self._voice_ids

    @property
    def voice_id(self) -> str:
        """
        Get the current voice ID for the text-to-speech engine.

        Returns
        -------
        str
            The current voice ID.
        """
        return self._voice_id

    @voice_id.setter
    def voice_id(self, voice_id: str) -> None:
        """
        Set the current voice ID for the text-to-speech engine.

        Parameters
        ----------
        voice_id : str
            The voice ID to set.
        """
        if voice_id in self._voice_ids:
            self._voice_id = voice_id
        else:
            self._logger.error(f"Invalid voice ID: {voice_id}. Options: {self._voice_ids}")

    @property
    def is_slow(self) -> bool:
        """
        Get whether the text-to-speech engine is set to speak slowly.

        Returns
        -------
        bool
            Whether the text-to-speech engine is set to speak slowly.
        """
        return self._is_slow

    @is_slow.setter
    def is_slow(self, is_slow: bool) -> None:
        """
        Set whether the text-to-speech engine is set to speak slowly.

        Parameters
        ----------
        is_slow : bool
            Whether to set the text-to-speech engine to speak slowly
        """
        self._is_slow = is_slow

    @abc.abstractmethod
    def say_async(self, text: str) -> None:
        """
        Speak the given text asynchronously.

        Parameters
        ----------
        text : str
            The text to speak.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_speaking(self) -> bool:
        """
        Return whether the text-to-speech engine is currently speaking.

        Returns
        -------
        bool
            Whether the text-to-speech engine is currently speaking.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def say(self, text: str) -> None:
        """
        Speak the given text synchronously.

        Parameters
        ----------
        text : str
            The text to speak.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self) -> None:
        """
        Stop speaking the current text.
        """
        raise NotImplementedError

    def is_file_type_supported(self, filepath: str) -> bool:
        """
        Checks whether the file type is supported by the text-to-speech engine.
        This is a static method to enforce that every text-to-speech engine
        supports the same file type(s). Currently, only MP3 is supported.

        Parameters
        ----------
        filepath : str
            The path of the file to check.

        Returns
        -------
        bool
            Whether the file type is supported.
        """
        exts = [".mp3"]
        for ext in exts:
            if filepath.lower().strip().endswith(ext):
                return True
        self._logger.error(f"Unsupported file type: {filepath}. Must end in {exts}")
        return False

    @abc.abstractmethod
    def save_to_file(self, text: str, filepath: str, **kwargs: Any):
        """
        Save the given text to an audio file.

        Parameters
        ----------
        text : str
            The text to save.
        filepath : str
            The path to save the audio file.
        """
        raise NotImplementedError
