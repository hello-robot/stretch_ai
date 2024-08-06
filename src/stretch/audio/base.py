import abc

import numpy as np


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
        pass

    @abc.abstractmethod
    def transcribe_file(self, audio_file: str) -> str:
        """Transcribe the audio file to text.

        Args:
            audio_file (str): Path to the audio file.

        Returns
            str: Transcribed text.
        """
        pass


class AbstractTextToSpeech(abc.ABC):
    """Basic text to speech module"""

    @abc.abstractmethod
    def speak(self, text: str):
        """Speak the given text.

        Args:
            text (str): The text to speak.
        """
        pass
