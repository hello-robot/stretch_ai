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

import numpy as np
import pyaudio
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
