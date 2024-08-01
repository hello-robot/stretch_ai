#!/usr/bin/env python3

"""
A simple audio recorder using PyAudio.
"""

import wave
from typing import List

import numpy as np
import pyaudio

from stretch.audio import AbstractSpeechToText


class AudioRecorder:
    """
    A class for recording audio from a microphone and saving it as a WAV file.
    """

    def __init__(
        self,
        chunk: int = 1024,
        channels: int = 2,
        sample_rate: int = 44100,
    ) -> None:
        """
        Initialize the AudioRecorder.

        Args:
            filename (str): Name of the output WAV file. Defaults to "recording.wav".
            chunk (int): Number of frames per buffer. Defaults to 1024.
            channels (int): Number of channels to record. Defaults to 2 (stereo).
            sample_rate (int): Sampling rate. Defaults to 44100 Hz.
        """
        self.chunk: int = chunk
        self.channels: int = channels
        self.sample_rate: int = sample_rate
        self.format: int = pyaudio.paInt16
        self.audio: pyaudio.PyAudio = pyaudio.PyAudio()

        # Stream object - used if you want to record audio in real-time
        self.stream: pyaudio.Stream = None

        self.reset()

    def reset(self) -> None:
        self.frames: List[bytes] = []
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()

    def start(self) -> None:
        """
        Start recording audio.
        """
        self.stream = self.get_stream()
        self.stream.start_stream()

    def record(self, filename: str = "recording.wav", duration: float = 5.0) -> None:
        """
        Record audio from the microphone for a specified duration.

        Args:
            duration (float): Recording duration in seconds.
        """
        stream: pyaudio.Stream = self.get_stream()

        print("Recording...")

        for _ in range(0, int(self.sample_rate / self.chunk * duration)):
            data: bytes = stream.read(self.chunk)
            self.frames.append(data)

        print("Recording finished.")

        stream.stop_stream()
        stream.close()

        self.audio.terminate()
        self.save(filename)
        self.reset()

    def get_stream(self) -> pyaudio.Stream:
        """Return an audio stream."""
        return self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

    def transcribe(self, text_to_speech: AbstractSpeechToText) -> None:
        """Use this audio stream"""
        while True:
            # Read audio data
            data = self.stream.read(self.chunk)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Convert audio data to float32 and normalize
            audio_float32 = audio_data.astype(np.float32) / 32768

            text = text_to_speech.process_audio(audio_float32)
            print(text)

    def save(self, filename: str) -> None:
        """
        Save the recorded audio as a WAV file.
        """
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(self.frames))

        print(f"Audio saved as {filename}")


if __name__ == "__main__":
    recorder = AudioRecorder()
    # Records 5 seconds of audio and saves it as "recording.wav"
    recorder.record("recording.wav", duration=5)
