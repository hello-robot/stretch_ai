# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os
from typing import List, Optional

from stretch.mapping.voxel.voxel_map_dynamem import SparseVoxelMap


class SparseVoxelMapEQA(SparseVoxelMap):
    def __init__(self, *args, **kwargs):
        # This field has to be set to True, as it allows the DynaMem to generate visual clues for each image
        kwargs["intelligent_exploration"] = True

        super().__init__(*args, **kwargs)

        # Cached question
        self._question: Optional[str] = None
        self.relevant_objects: Optional[list] = None

        self.history_outputs: List[str] = []

    def extract_relevant_objects(self, question: str):
        """
        Parsed the question and extract few keywords for DynaMem voxel map to select relevant images
        Example:
                Is there grey cloth on cloth hanger?
                gery cloth,cloth hanger
        """
        if self._question != question:
            self._question = question
            # The cached question is not the same as the question provided
            prompt = """
                Assume there is an agent doing Question Answering in an environment.
                When it receives a question, you need to tell the agent few objects (preferably 1-3) it needs to pay special attention to.
                Example:
                    Where is the pen?
                    pen

                    Is there grey cloth on cloth hanger?
                    gery cloth,cloth hanger
            """
            messages = [prompt, self._question]
            # To avoid initializing too many clients and using up too much memory, I reused the client generating the image descriptions even though it is a VL model
            self.relevant_objects = self.image_description_client(messages).split(",")
            print("relevant objects to look at", self.relevant_objects)
            self.history_outputs = []

    def log_text(self, commands):
        """
        Log the text input and image input into some files for debugging and visualization
        """
        if not os.path.exists(self.log + "/" + str(len(self.image_descriptions))):
            os.makedirs(self.log + "/" + str(len(self.image_descriptions)))
            input_texts = ""
            for command in commands:
                input_texts += command + "\n"
            with open(
                self.log + "/" + str(len(self.image_descriptions)) + "/input.txt", "w"
            ) as file:
                file.write(input_texts)

    def parse_answer(self, answer_outputs: str):

        """
        Parse the output of LLM text into reasoning, answer, confidence, action, confidence_reasoning
        """

        # Log LLM output
        with open(self.log + "/" + str(len(self.image_descriptions)) + "/output.txt", "w") as file:
            file.write(answer_outputs)

        # Answer outputs in the format "Caption: Reasoning: Answer: Confidence: Action: Confidence_reasoning:"
        def extract_between(text, start, end):
            try:
                return (
                    text.split(start, 1)[1]
                    .split(end, 1)[0]
                    .strip()
                    .replace("\n", "")
                    .replace("\t", "")
                )
            except IndexError:
                return ""

        def extract_after(text, start):
            try:
                return text.split(start, 1)[1].strip().replace("\n", "").replace("\t", "")
            except IndexError:
                return ""

        reasoning = extract_between(answer_outputs, "reasoning:", "answer:")
        answer = extract_between(answer_outputs, "answer:", "confidence:")
        confidence_text = extract_between(answer_outputs, "confidence:", "action:")
        confidence = "true" in confidence_text.replace(" ", "")
        action = extract_between(answer_outputs, "action:", "confidence_reasoning:")
        confidence_reasoning = extract_after(answer_outputs, "confidence_reasoning:")

        return reasoning, answer, confidence, action, confidence_reasoning
