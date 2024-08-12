# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import numpy as np
import whisper

from .base import AbstractSpeechToText


class WhisperSpeechToText(AbstractSpeechToText):
    """Use the whisper model to transcribe speech to text."""

    def __init__(self):
        self.model = whisper.load_model("base")

    def transcribe_file(self, audio_file: str) -> str:
        """Transcribe the audio file to text."""
        result = self.model.transcribe(audio_file)
        return result["text"]

    def transcribe_stream(self, stream, chunk: int = 1024) -> str:
        """Transcribe the audio stream to text."""
        while True:
            # Read audio data
            data = stream.read(chunk)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Convert audio data to float32 and normalize
            audio_float32 = audio_data.astype(np.float32) / 32768.0

            # Transcribe the audio chunk
            result = self.model.transcribe(audio_float32)
            print(result["text"])

        stream.stop_stream()
        stream.close()

    def process_audio(self, audio_data: np.ndarray) -> str:
        """Process audio data."""
        result = self.model.transcribe(audio_data)
        return result["text"]


if __name__ == "__main__":
    wst = WhisperSpeechToText()
    print(wst.transcribe_file("recording.wav"))

    # Record audio
    # from .audio_recorder import AudioRecorder

    # audio_recorder = AudioRecorder()

    # Get stream
    # stream = audio_recorder.get_stream()

    # Transcribe in a loop
    # wst.transcribe_stream(stream)
