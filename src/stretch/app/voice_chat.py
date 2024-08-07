# Copyright (c) Hello Robot, Inc.
#
# This source code is licensed under the APACHE 2.0 license found in the
# LICENSE file in the root directory of this source tree.
# 
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.


import os
import tempfile
import timeit

import click
import torch
from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from stretch.audio import AudioRecorder
from stretch.audio.speech_to_text import WhisperSpeechToText
from stretch.llms import Gemma2bClient, LlamaClient
from stretch.llms.prompts.simple_prompt import SimpleStretchPromptBuilder


@click.command()
@click.option(
    "--model",
    default="gemma",
    help="The model to use (gemma or llama)",
    type=click.Choice(["gemma", "llama"]),
)
@click.option(
    "--max_audio_duration", default=10.0, help="The maximum duration of the audio recording"
)
@click.option(
    "--silence_limit", default=2.0, help="The amount of silence before stopping the recording"
)
def main(model="gemma", max_audio_duration: float = 10.0, silence_limit: float = 2.0):
    # Load the tokenizer and model
    audio_recorder = AudioRecorder()
    whisper = WhisperSpeechToText()
    prompt = SimpleStretchPromptBuilder()
    if model == "gemma":
        client = Gemma2bClient(prompt)
    elif model == "llama":
        client = LlamaClient(prompt)
    else:
        raise ValueError(f"Invalid model: {model}")

    print("Talk to me, Stretch! If you don't say anything, I will give up.")
    for i in range(50):
        # Record audio
        input(colored("Press enter to speak or ctrl+c to exit.", "yellow"))

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            temp_filename = temp_audio_file.name
            audio_recorder.record(
                temp_filename, duration=max_audio_duration, silence_limit=silence_limit
            )

            # Transcribe the audio file
            input_text = whisper.transcribe_file(temp_filename)

        # Remove the temporary file
        os.remove(temp_filename)

        if len(input_text) == 0:
            break

        # Get the response and time it
        t0 = timeit.default_timer()
        assistant_response = client(input_text)
        t1 = timeit.default_timer()

        # Decode and print the result
        print(colored("I heard:", "green"), input_text)
        print(colored("Response:", "blue"), assistant_response)
        print("-" * 80)
        print("Time taken:", t1 - t0)
        print("-" * 80)


if __name__ == "__main__":
    main()
