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

import click
from termcolor import colored

from stretch.audio import AudioRecorder
from stretch.audio.speech_to_text import WhisperSpeechToText
from stretch.llms import get_llm_choices, get_llm_client, get_prompt_builder, get_prompt_choices


@click.command()
@click.option(
    "--llm",
    default="gemma2b",
    help="The model to use (gemma or llama)",
    type=click.Choice(get_llm_choices()),
)
@click.option(
    "--prompt", default="simple", help="The prompt to use", type=click.Choice(get_prompt_choices())
)
@click.option(
    "--max_audio_duration", default=10.0, help="The maximum duration of the audio recording"
)
@click.option(
    "--silence_limit", default=2.0, help="The amount of silence before stopping the recording"
)
@click.option("--voice", default=False, help="Enable voice chat", is_flag=True)
def main(
    llm="gemma2b",
    max_audio_duration: float = 10.0,
    silence_limit: float = 2.0,
    voice=False,
    prompt="simple",
):
    prompt = get_prompt_builder(prompt)
    client = get_llm_client(llm, prompt)

    if voice:
        # Load the tokenizer and model
        audio_recorder = AudioRecorder()
        whisper = WhisperSpeechToText()
    else:
        audio_recorder = None
        whisper = None

    if voice:
        print("Talk to me, Stretch! If you don't say anything, I will give up.")
    for i in range(50):
        if voice:
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

                print(colored("I heard:", "green"), input_text)
        else:
            input_text = input(colored("You: ", "green"))

        if len(input_text) == 0:
            break

        # Get the response and time it
        t0 = timeit.default_timer()
        assistant_response = client(input_text)
        t1 = timeit.default_timer()

        # Decode and print the result
        print(colored("Response:", "blue"), prompt.parse_response(assistant_response))
        print("-" * 80)
        print("Time taken:", t1 - t0)
        print("-" * 80)


if __name__ == "__main__":
    main()
