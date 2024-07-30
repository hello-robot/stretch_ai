import whisper

from .base import AbstractSpeechToText


class WhisperSpeechToText(AbstractSpeechToText):
    """Use the whisper model to transcribe speech to text."""

    def __init__(self):
        self.model = whisper.load_model("base")

    def transcribe(self, audio_file: str) -> str:
        """Transcribe the audio file to text."""
        result = self.model.transcribe(audio_file)
        return result["text"]


if __name__ == "__main__":
    wst = WhisperSpeechToText()
    print(wst.transcribe("test.wav"))
