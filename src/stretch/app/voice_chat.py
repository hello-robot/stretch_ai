import torch
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

import timeit

from transformers import pipeline


def main():
    # Load the tokenizer and model
    audio_recorder = AudioRecorder()
    whisper = WhisperSpeechToText()
    pipe = pipeline(
        "text-generation",
        model="google/gemma-2-2b-it",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",  # replace with "mps" to run on a Mac device
    )

    conversation_history = []

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
            new_message = {"role": "user", "content": prompt + input_text}
        else:
            new_message = {"role": "user", "content": input_text}

        conversation_history.append(new_message)

        # Prepare the messages including the conversation history
        messages = conversation_history.copy()

        t0 = timeit.default_timer()
        outputs = pipe(messages, max_new_tokens=512)
        t1 = timeit.default_timer()

        assistant_response = outputs[0]["generated_text"][-1]["content"].strip()

        # Add the assistant's response to the conversation history
        conversation_history.append({"role": "assistant", "content": assistant_response})

        # Decode and print the result
        print(colored("I heard:", "green"), input_text)
        print(colored("Response:", "blue"), assistant_response)
        print("-" * 80)
        print("Time taken:", t1 - t0)
        print("-" * 80)


if __name__ == "__main__":
    main()
