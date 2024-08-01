#!/usr/bin/env python3

"""
A simple audio recorder using PyAudio.
"""

import wave
from typing import List

import pyaudio


class AudioRecorder:
    """
    A class for recording audio from a microphone and saving it as a WAV file.
    """

    def __init__(
        self,
        filename: str = "recording.wav",
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
        self.filename: str = filename
        self.chunk: int = chunk
        self.channels: int = channels
        self.sample_rate: int = sample_rate
        self.format: int = pyaudio.paInt16
        self.audio: pyaudio.PyAudio = pyaudio.PyAudio()
        self.frames: List[bytes] = []

    def record(self, duration: float) -> None:
        """
        Record audio from the microphone for a specified duration.

        Args:
            duration (float): Recording duration in seconds.
        """
        stream: pyaudio.Stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

        print("Recording...")

        for _ in range(0, int(self.sample_rate / self.chunk * duration)):
            data: bytes = stream.read(self.chunk)
            self.frames.append(data)

        print("Recording finished.")

        stream.stop_stream()
        stream.close()

    def save(self) -> None:
        """
        Save the recorded audio as a WAV file.
        """
        self.audio.terminate()

        with wave.open(self.filename, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(self.frames))

        print(f"Audio saved as {self.filename}")


if __name__ == "__main__":
    recorder = AudioRecorder()
    # Records 5 seconds of audio
    recorder.record(5)
    # Saves it to file
    recorder.save()
