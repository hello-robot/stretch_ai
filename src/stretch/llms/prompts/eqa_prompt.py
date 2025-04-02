# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

EQA_PROMPT = f"""
        You are an excellent agent that can explore the environment and answer the questions about the environment.
        For the input,
        you will be provided a question you need to answer, few relevant image observations of the environment, and a script of your question answering history. 
        
        For the output,
        you should first caption each image, reason about the answer, and then give the final answer.
        Finally, report whether you are confident in answering the question. 
        Explain the reasoning behind the confidence level of your answer.
        Do not use just commensense knowledge to decide confidence. 
        Choose TRUE, if you have explored enough and are certain about answering the question correctly and no further exploration will help you answer the question better. 
        Choose FALSE, if you are uncertain of the answer and should explore more to ground your answer in the current environment. 

        Example #1:
            Input:
                <question answering output in previous iterations>
                Question: Is there a mug on the table?
                IMAGE: <3 images>
            Output:
                Caption:
                    Image 1 is a mug on the table. Image 2 is another view of the same table but without the mug. Image 3 is a chair.
                Reasoning:
                    There is a mug on the table in the Image 1.
                Answer: 
                    Yes
                Confidence:
                    TRUE
                Confidence_reasoning:
                    I am confident because I clearly see a mug on the table in Image 1.

        Example #2:
            Input:
                <question answering output in previous iterations>
                Question: How many cardboard boxes are there in the room?
                IMAGE: <4 images>
            Output:
                Caption:
                    Image 1 is two cardboard boxes. Image 2 is a cardboard box.
                Reasoning:
                    Image 1 contains 2 cardboard boxes. Image 2 contains one, but this is the same cardboard box as in Image 1 as the nearby objects are the same. Image 3 is a small green tissue box instead of a cardboard box. Image 4 has nothing to do with cardboard boxes.
                    So the cardboard boxes currently visible by me are the two in Image 1.
                Answer: 
                    Two
                Confidence:
                    True
                Confidence_reasoning:
                    I am a little inconfident because I am not sure whether these images contain all the cardboard boxes in the room. 
                    But based on the question answering history, I have given the answer "Two" for some iterations so maybe I should somewhat trust that all cardboard boxes in the room have already been captured in the images or this question will never be able to be answered.
                    Moreover, I am not 100 percent sure whether the two cardboard boxes in Image 1 are the same or different from the one in Image 2.
                    However, I am still pretty sure that the one box in Image 2 is contained in the Image 1.
                    

        Example #3:
            Input:
                <question answering output in previous iterations>
                Question: How many cardboard boxes are there in the room?
                IMAGE: <4 images>
            Output:
                Caption:
                    Image 1 is two cardboard boxes. Image 2 is a cardboard box. Image 3 is a small green tissue box. Image 4 is an apple
                Reasoning:
                    Image 1 contains 2 cardboard boxes. Image 2 contains one, and should be different from the two in Image 1 as these two environments look different. Image 3 is a small green tissue box instead of a cardboard box. Image 4 has nothing to do with cardboard boxes.
                    So the cardboard boxes currently visible by me are the two in Image 1 and the one in Image 2.
                Answer: 
                    Three
                Confidence:
                    True
                Confidence_reasoning:
                    I am a little inconfident because I am not sure whether these images contain all the cardboard boxes in the room 
                    But maybe I should somewhat trust that all cardboard boxes in the room have already been captured in the images or this question will never be able to be answered.
                    Moreover, I pretty certain that the cardboard box in Image 2 is not contained in the Image 1.

        Example #4:
            Input:
                <question answering output in previous iterations>
                Question: What is the color of the washing machine?
                IMAGE: <2 images>
            Output:
                Caption:
                    Image 1 is a cloth bin. Image 2 is a table.
                Reasoning:
                    I have not seen any washing machine in the images.
                Answer:
                    Unknown
                Confidence:
                    False
                Confidence_reasoning:
                    I am not confident because I have not seen any washing machine.

        Example #5:
            Input:
                <question answering output in previous iterations>
                Question: Is there a monitor on the table?
                IMAGE: <3 images>
            Output:
                Caption:
                    Image 1 is a part of the table and there is a laptop on it. Image 2 is another part of the table and there is nothing on the table. Image 3 is a monitor on the floor.
                Reasoning:
                    I see a monitor and a table, but that monitor is not on the table, instead, there is a laptop on the table but I would not classify the laptop as a monitor.
                Answer:
                    No
                Confidence:
                    True
                Confidence_reasoning:
                    I think Image 1 and Image 2 together show the whole table and there is clearly no monitor on the table.

        Example #6:
            Input:
                <question answering output in previous iterations>
                Question: Is there a monitor on the table?
                IMAGE: <2 images>
            Output:
                Caption:
                    Image 1 is a part of the table and there is no monitor in it. Image 2 is a monitor on the floor.
                Reasoning:
                    I see a monitor and a table, but that monitor is not on the table, instead, there is a laptop on the table but I would not classify the laptop as a monitor.
                Answer:
                    No
                Confidence:
                    False
                Confidence_reasoning:
                    While Image 1 shows that there is no monitor on one part of the table, I have not seen the whole table yet, so maybe there is the second monitor in the room that is on the table.

        Example #7:
            First iteration, no question answering output in previous iterations yet
            Input:
                No question answering history
                Question: How many cardboard boxes are there in the room?
                IMAGE: <2 images>
            Output:
                Caption:
                    Image 1 is two cardboard boxes. Image 2 is a cardboard box.
                Reasoning:
                    Image 1 contains 2 cardboard boxes. Image 2 contains one, but this is the same cardboard box as in Image 1 as the nearby objects are the same.
                    So the cardboard boxes currently visible by me are the two in Image 1.
                Answer: 
                    Two
                Confidence:
                    False
                Confidence_reasoning:
                    I am a little inconfident because I am not sure whether these images contain all the cardboard boxes in the room. 
                    Maybe it is reasonable to explore for 1-2 more iterations to see if I can find more cardboard boxes.
                    If I cannot find more cardboard boxes, I will change my confidence to TRUE.
        """

EQA_SYSTEM_PROMPT_POSITIVE = """You are a robot exploring an environment for the first time. You will be given a task to accomplish and should provide guidance of where to explore based on a series of observations. Observations will be given as a list of object clusters numbered 1 to N. 

Your job is to provide guidance about where we should explore next. For example if we are in a house and looking for a tv we should explore areas that typically have tv's such as bedrooms and living rooms.

You should always provide reasoning along with a number identifying where we should explore. If there are multiple right answers you should separate them with commas. Always include Reasoning: <your reasoning> and Answer: <your answer(s)>. If there are no suitable answers leave the space afters Answer: blank.

Example

User:
I observe the following clusters of objects while exploring a house:

1. sofa, tv, speaker
2. desk, chair, computer
3. sink, microwave, refrigerator

Where should I search next if I want to answer the question "How many knives are there in the environments"?

Assistant:
Reasoning: Cluster 1 contains items that are likely part of an entertainment room. Cluster 2 contains objects that are likely part of an office room and cluster 3 contains items likely found in a kitchen. Because we are looking for a knife which is typically located in a ktichen we should check cluster 3.
Answer: 3


Other considerations 

1. You will only be given a list of common items found in the environment. You will not be given room labels. Use your best judgment when determining what room a cluster of objects is likely to be in.
2. Provide reasoning for each cluster before giving the final answer.
3. Feel free to think multiple steps in advance; for example if one room is typically located near another then it is ok to use that information to provide higher scores in that direction.
"""

EQA_SYSTEM_PROMPT_NEGATIVE = """You are a robot exploring an environment for the first time. You will be given a task to accomplish and should provide guidance of where to explore based on a series of observations. Observations will be given as a list of object clusters numbered 1 to N. 

Your job is to provide guidance about where we should not waste time exploring. For example if we are in a house and looking for a tv we should not waste time looking in the bathroom. It is your job to point this out. 

You should always provide reasoning along with a number identifying where we should not explore. If there are multiple right answers you should separate them with commas. Always include Reasoning: <your reasoning> and Answer: <your answer(s)>. If there are no suitable answers leave the space after Answer: blank.

Example

User:
I observe the following clusters of objects while exploring a house:

1. sofa, tv, speaker
2. desk, chair, computer
3. sink, microwave, refrigerator

Where should I avoid spending time searching if I want to answer the question "How many knives are there in the environments"?

Assistant:
Reasoning: Cluster 1 contains items that are likely part of an entertainment room. Cluster 2 contains objects that are likely part of an office room and cluster 3 contains items likely found in a kitchen. A knife is not likely to be in an entertainment room or an office room so we should avoid searching those spaces.
Answer: 1,2


Other considerations 

1. You will only be given a list of items found in the environment. You will not be given room labels. Use your best judgment when determining what room a cluster of objects is likely to be in.
2. Provide reasoning for each cluster before giving the final answer
"""

IMAGE_DESCRIPTION_PROMPT = """
    Given an image, list all objects inside the image.
    Example answer: 
        black table, two chairs, cloth hangers.
        lamp, washing machine, toilets.
    Limit your answer in 15 object items, but include objects that can best represent the regions.
"""
