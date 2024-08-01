#!/usr/bin/env python3

"""
A simple audio recorder using PyAudio.
"""

import audioop
import wave
from typing import List

import numpy as np
import pyaudio
from tqdm import tqdm

from stretch.audio.base import AbstractSpeechToText


class AudioRecorder:
    """
    A class for recording audio from a microphone and saving it as a WAV file.
    """

    def __init__(
        self,
        chunk: int = 1024,
        channels: int = 2,
        sample_rate: int = 44100,
        volume_threshold: int = 500,
    ) -> None:
        """
        Initialize the AudioRecorder.

        Args:
            filename (str): Name of the output WAV file. Defaults to "recording.wav".
            chunk (int): Number of frames per buffer. Defaults to 1024.
            channels (int): Number of channels to record. Defaults to 2 (stereo).
            sample_rate (int): Sampling rate. Defaults to 44100 Hz.
            volume_threshold (int): Minimum volume threshold to start recording. Defaults to 500.
        """
        self.chunk: int = chunk
        self.channels: int = channels
        self.sample_rate: int = sample_rate
        self.format: int = pyaudio.paInt16
        self.volume_threshold: int = volume_threshold
        self.audio: pyaudio.PyAudio = pyaudio.PyAudio()

        # Stream object - used if you want to record audio in real-time
        self.stream: pyaudio.Stream = None

        self.reset()
        self.start()

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

    def record(
        self, filename: str = "recording.wav", duration: float = 10.0, silence_limit: float = 1.0
    ) -> None:
        """
        Record audio from the microphone for a specified duration.

        Args:
            duration (float): Recording duration in seconds.
        """

        # Tracks if we have started hearing things
        audio_started: bool = False
        silent_chunks: int = 0

        print("Recording...")

        for _ in tqdm(range(0, int(self.sample_rate / self.chunk * duration))):

            data: bytes = self.stream.read(self.chunk)
            self.frames.append(data)

            rms = audioop.rms(data, 2)  # Get audio level

            if audio_started:
                self.frames.append(data)

            if rms > self.volume_threshold:
                audio_started = True
                silent_chunks = 0
            elif audio_started:
                silent_chunks += 1

            if silent_chunks > silence_limit * self.sample_rate / self.chunk:
                break

        print("Recording finished.")

        self.audio.terminate()
        self.save(filename)
        self.reset()

    def __del__(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

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
