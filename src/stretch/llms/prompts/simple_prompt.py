from stretch.llms.base import AbstractPromptBuilder

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


class SimplePromptBuilder(AbstractPromptBuilder):
    def __init__(self, prompt: str):
        self.prompt_str = prompt

    def __str__(self):
        return self.prompt_str

    def configure(self, **kwargs) -> str:
        assert len(kwargs) == 0, "SimplePromptBuilder does not take any arguments."
        return self.prompt_str


class SimpleStretchPromptBuilder(AbstractPromptBuilder):
    def __init__(self):
        self.prompt_str = simple_stretch_prompt

    def __str__(self):
        return self.prompt_str

    def configure(self, **kwargs) -> str:
        assert len(kwargs) == 0, "SimpleStretchPromptBuilder does not take any arguments."
        return self.prompt_str
