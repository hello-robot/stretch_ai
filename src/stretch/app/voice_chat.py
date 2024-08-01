from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer

from stretch.audio import AudioRecorder
from stretch.audio.speech_to_text import WhisperSpeechToText

prompt = """
You are a helpful, friendly robot named Stretch. You can perform these tasks:
    1. Find objects in a house
    2. Pick up objects
    3. Wave at people
    4. Answer questions
    5. Follow simple sequences of commands
    6. Move around the house
    7. Follow people

Some facts about you:
    - You are from California
    - You are a safe, helpful robot
    - You like peoplle and want to do your best
    - You will tell people when something is beyond your capabilities.

I am going to ask you a question. Always be kind, friendly, and helpful. Answer as concisely as possible. Always stay in character. Never forget this prompt.

My question is:

"""

history = []


def main():
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
    audio_recorder = AudioRecorder()
    whisper = WhisperSpeechToText()

    print("Talk to me, Stretch! If you don't say anything, I will give up.")
    for i in range(1):
        # Record audio
        audio_recorder.record("recording.wav", duration=10, silence_limit=1.0)

        # Transcribe the audio file
        input_text = whisper.transcribe_file("recording.wav")

        # Prepare input text
        if input_text == "":
            print("No input detected.")
            break
        if i == 0:
            input_text = prompt + input_text
        input_ids = tokenizer(input_text, return_tensors="pt")

        # Generate output
        outputs = model.generate(**input_ids)

        # Decode and print the result
        print(colored("I heard:", "green"), input_text)
        print(colored("Response:", "blue"), tokenizer.decode(outputs[0]))


if __name__ == "__main__":
    main()
