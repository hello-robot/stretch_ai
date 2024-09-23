# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os
import tempfile
import timeit
from typing import Any

from termcolor import colored

from stretch.audio import AudioRecorder
from stretch.audio.speech_to_text import WhisperSpeechToText
from stretch.llms import AbstractLLMClient, AbstractPromptBuilder


class LLMChatWrapper:
    """Simple class that will get user information and return prompts."""

    def __init__(
        self,
        llm_client: AbstractLLMClient,
        prompt: AbstractPromptBuilder,
        voice: bool = False,
        max_audio_duration: int = 10,
        silence_limit: int = 2,
    ):
        self.voice = voice
        self.llm_client = llm_client
        self.prompt = prompt
        self.max_audio_duration = max_audio_duration
        self.silence_limit = silence_limit

        if self.voice:
            # Load the tokenizer and model
            self.audio_recorder = AudioRecorder()
            self.whisper = WhisperSpeechToText()
        else:
            audio_recorder = None
            whisper = None

    def query(self, verbose: bool = False) -> Any:
        if self.voice:
            print("-" * 80)
            input(colored("Press enter to speak or ctrl+c to exit.", "yellow"))
            print("-" * 80)

            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                temp_filename = temp_audio_file.name
                self.audio_recorder.record(
                    temp_filename,
                    duration=self.max_audio_duration,
                    silence_limit=self.silence_limit,
                )

                # Transcribe the audio file
                input_text = self.whisper.transcribe_file(temp_filename)

                # Remove the temporary file
                os.remove(temp_filename)

                print(colored("I heard:", "green"), input_text)
        else:
            print("-" * 80)
            input_text = input(colored("You: ", "green"))
            print("-" * 80)

        if len(input_text) == 0:
            return None

        # Get the response and time it
        t0 = timeit.default_timer()
        assistant_response = self.llm_client(input_text)
        t1 = timeit.default_timer()

        response = self.prompt.parse_response(assistant_response)

        if verbose:
            # Decode and print the result
            print(colored("Response:", "blue"), response)
            print("-" * 80)
            print("Time taken:", t1 - t0)
            print("-" * 80)

        return response

    def say(self, text: str) -> None:
        if self.voice:
            raise NotImplementedError("Voice is not supported yet.")
        else:
            print("-" * 80)
            print(colored("Robot:", "blue"), text)
            print("-" * 80)
