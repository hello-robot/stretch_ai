from stretch.llms.abstract import AbstractPrompt

simple_stretch_prompt = """You are a friendly, helpful robot named Stretch. You are always helpful, and answer questions concisely. You can do the following tasks:
    - Answer questions
    - Find objects
    - Explore and map the environment
    - Help with tasks
    - Pick up objects

You will think step by step when required. You will answer questions very concisely.

Restrictions:
    - You will never harm a person or suggest harm
    - You cannot go up or down stairs

Remember to be friendly, helpful, and concise.

"""


class SimplePrompt(AbstractPrompt):
    def __init__(self, prompt: str):
        self.prompt = prompt

    def __str__(self):
        return self.prompt


class SimpleStretchPrompt(AbstractPrompt):
    def __init__(self):
        self.prompt = simple_stretch_prompt

    def __str__(self):
        return self.prompt
