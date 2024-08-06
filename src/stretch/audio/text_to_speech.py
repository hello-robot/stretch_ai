import os

import yaml
from google.cloud import texttospeech

from stretch.utils.config import get_full_config_path

from .base import AbstractTextToSpeech


class GoogleCloudTextToSpeech(AbstractTextToSpeech):
    def __init__(self):
        # Create a client
        self.client = texttospeech.TextToSpeechClient()

    def play_sound_file(self, sound_file: str):
        os.system(f"aplay --nonblock {sound_file}")

    def speak(self, text: str):
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Construct the voice request
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.MALE,
            name="en-US-Neural2-I",
        )

        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)

        response = self.client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Write the response to an output file and play the file
        sound_file = "/tmp/tts_output.mp3"

        with open(sound_file, "wb+") as out:
            # Write the response to the output file.
            out.write(response.audio_content)
            print(f'Audio content written to file "{sound_file}"')

        self.play_sound_file(sound_file)
