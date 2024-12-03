# How to Add a New LLM Task

With Stretch AI, you can command your robot to perform tasks using speech and a large language model (LLM). This documentation provides an example of adding a new task to Stretch AI that can be called by an LLM. Specifically, it describes how we added a **simple** handover task to the [pick-and-place demo](https://github.com/hello-robot/stretch_ai/blob/main/docs/llm_agent.md) that comes with Stretch AI. 

As a reminder, you can run this demo, which includes the handover task, using the following command: 

```
python -m stretch.app.ai_pickup --use_llm --use_voice
```


## Overview

Creating the new handover task primarily involved the following steps:

- **[Create a New Task](#create-a-new-task)**
  - [hand_over_task.py](../src/stretch/agent/task/pickup/hand_over_task.py) defines the new task. It's found in the [/src/stretch/agent/task/pickup](../src/stretch/agent/task/pickup) directory. 
- **[Create New Operations](#create-new-operations)** 
  - The handover task uses a series of operations, some of which we created specifically for the handover task. For example, the [extend_arm.py](../src/stretch/agent/operations/extend_arm.py) operation, found in the [/src/stretch/agent/operations](../src/stretch/agent/operations) directory, extends the arm during a handover.
- **[Create a New App](#create-a-new-app)**
  - The [hand_over_object.py](../src/stretch/app/hand_over_object.py) app, found in the [/src/stretch/app/](../src/stretch/app) directory, tests the new handover task in isolation, which can simplify development. 
- **[Update the Executor](#update-the-executor)**
  - The [PickupExecutor](https://github.com/hello-robot/stretch_ai/blob/64c718773bad384599752ce6f52e6add9013b92d/src/stretch/agent/task/pickup/pickup_executor.py#L27) processes a list of task tuples resulting from the use of an LLM and executes the associated tasks. We added the [_hand_over](https://github.com/hello-robot/stretch_ai/blob/64c718773bad384599752ce6f52e6add9013b92d/src/stretch/agent/task/pickup/pickup_executor.py#L172) method and the [conditional statement to call it](https://github.com/hello-robot/stretch_ai/blob/64c718773bad384599752ce6f52e6add9013b92d/src/stretch/agent/task/pickup/pickup_executor.py#L246) to the PickupExecutor in [pickup_executor.py](../src/stretch/agent/task/pickup/pickup_executor.py).
- **[Modify the LLM Prompt](#modify-the-llm-prompt)**
  - Modifying the LLM prompt enables the LLM to call the new handover task. Specifically, we edited the [LLM prompt text](https://github.com/hello-robot/stretch_ai/blob/64c718773bad384599752ce6f52e6add9013b92d/src/stretch/llms/prompts/pickup_prompt.py#L79) and code in [pickup_prompt.py](../src/stretch/llms/prompts/pickup_prompt.py), which is found in the [/src/stretch/llms/prompts](../src/stretch/llms/prompts) directory.

We'll now provide details for each of these steps.

## Create A New Task

We started by copying an existing task and modifying it. This mostly involved editing the [get_one_shot_task](https://github.com/hello-robot/stretch_ai/blob/64c718773bad384599752ce6f52e6add9013b92d/src/stretch/agent/task/pickup/hand_over_task.py#L59) method. The get_one_shot_task method first [creates a task](https://github.com/hello-robot/stretch_ai/blob/64c718773bad384599752ce6f52e6add9013b92d/src/stretch/agent/task/pickup/hand_over_task.py#L62).

`task = Task()`

It then creates operations and [adds them to the task](https://github.com/hello-robot/stretch_ai/blob/64c718773bad384599752ce6f52e6add9013b92d/src/stretch/agent/task/pickup/hand_over_task.py#L146). In total, the handover task performs nine operations in a simple linear sequence. 

```
# rotate the head and look for a person
task.add_operation(update)

# announce that the robot has found a person
task.add_operation(found_a_person)

# navigate to the person
task.add_operation(go_to_object)

# announce that the robot is going to extend its arm
task.add_operation(ready_to_extend_arm)

# extend the arm
task.add_operation(extend_arm)

# announce that the robot is going to open its gripper
task.add_operation(ready_to_open_gripper)

# open the gripper
task.add_operation(open_gripper)

# announce that the robot has finished the task
task.add_operation(finished_handover)

# retract the arm
task.add_operation(retract_arm)
```

### Finding and Navigating to Objects

The most critical operations used by the handover task are the update operation and the go_to_object operation. update is an [UpdateOperation](../src/stretch/agent/operations/update.py) and go_to_object is a [NavigateToObjectOperation](../src/stretch/agent/operations/navigate.py).

An UpdateOperation first updates the robot's map. Then it looks for an instance in the map that matches the target_object using the match_method. **When the robot detects a significant object, it creates an instance for the object that gathers information about the detected object, such as a cuboid defined with respect to the map, point clouds, images, and feature vectors.** Stretch AI's  [Rerun](https://rerun.io/) visualization displays instances as colorful line drawn cuboids. You can learn more about instances from the code found in the [/src/stretch/mapping/instance](../src/stretch/mapping/instance) directory. For example, Instance and InstanceView are defined in [./instance/core.py](../src/stretch/mapping/instance/core.py).

The following code excerpt illustrates creating the update operation and configuring it to find a person by tilting the robot's head upward, rotating it around, and updating the map. After updating the map, it attempts to find an instance with a detected class name that exactly matches the string "person". 

```
update = UpdateOperation("update_scene", self.agent, retry_on_failure=True)

update.configure(
    ...
    target_object="person",
    match_method="name",
    move_head=True,
    tilt=-1.0 * np.pi / 8.0,
    ...
)
```

The NavigateToObjectOperation navigates to the current target object, which is set by the UpdateOperation.

### Detecting Relevant Objects

Notably, many of the steps in the pick and place demo use the "feature" match method to find instances. The "feature" match method uses embeddings to match text to images of the instance. We did not have success using the "feature" match method to find a person instance, so we instead used the original class label assigned by the underlying object detector that results in the creation of instances. 

**By default, the robot only creates instances for object categories listed in [example_cat_map.json](../src/stretch/config/example_cat_map.json) and detected by [Detic](https://github.com/facebookresearch/Detic).** Detic was trained with the [twenty-one-thousand classes](https://huggingface.co/datasets/huggingface/label-files/raw/main/imagenet-22k-id2label.json) used in the [ImageNet](https://en.wikipedia.org/wiki/ImageNet) database. By default, Stretch AI uses a relatively small number of classes (e.g., 108 classes when the handover task was originally created).

**To detect people for the handover task, we [added the "person" category](https://github.com/hello-robot/stretch_ai/blob/4be200f8ffa908bfe23b55139a8341916c4342f4/src/stretch/config/example_cat_map.json#L111) to example_cat_map.json.**

### Provide Audible Feedback with the Robot's Voice

We recommend that you have the robot provide frequent feedback via speech. For example, the handover task has the robot make four announcements, each of which uses the SpeakOperation class found in [speak.py](../src/stretch/agent/operations/speak.py). 

The first announcement creates the found_a_person operation with the following code:

```
found_a_person = SpeakOperation(...)

found_a_person.configure(
    message="I found a person! I am going to navigate to them.", 
    sleep_time=2.0
)
```

## Create New Operations

The available operations can be found in the [/src/stretch/agent/operations](../src/stretch/agent/operations) directory. We recommend that you use existing operations when possible. 

For the handover task, we defined new operations to perform the actual handover. For example, the [extend_arm.py](../src/stretch/agent/operations/extend_arm.py) operation, found in the [/src/stretch/agent/operations](../src/stretch/agent/operations) directory, extends the arm during a handover.

Like other operations, the ExtendArm operation class has a configure method and a run method. The configure method sets relevant parameters for the operation ahead of time. The task calls the run method to execute the operation. 

When defining an operation, it is good practice to check that the operation can be used before or after other common operations. For example, it's important to ensure that the robot is in the correct mode (i.e., manipulation mode or navigation mode) prior to commanding the robot. 

**After creating a new operation, be sure to update [src/stretch/agent/operations/__init__.py](../src/stretch/agent/operations/__init__.py), so that your new operations can be imported into your task code.** 

## Create A New App

When developing the handover task, we used an app to test it in isolation. This is not necessary, but can be convenient. 

We created the [hand_over_object.py](../src/stretch/app/hand_over_object.py) app, found in the [/src/stretch/app/](../src/stretch/app) directory. To try out the handover task in isolation without the robot holding an object, use the following command.

```
python -m stretch.app.hand_over_object --target_object "person"
```

## Update the Executor

Stretch AI uses an executor to process a list of task tuples and execute the relevant tasks. 

Since we wanted to add the handover task to the existing pick and place demo, we added the [_hand_over](https://github.com/hello-robot/stretch_ai/blob/64c718773bad384599752ce6f52e6add9013b92d/src/stretch/agent/task/pickup/pickup_executor.py#L172) method and the [conditional statement to call it](https://github.com/hello-robot/stretch_ai/blob/64c718773bad384599752ce6f52e6add9013b92d/src/stretch/agent/task/pickup/pickup_executor.py#L246) to the [PickupExecutor](https://github.com/hello-robot/stretch_ai/blob/64c718773bad384599752ce6f52e6add9013b92d/src/stretch/agent/task/pickup/pickup_executor.py#L27) class found in [pickup_executor.py](../src/stretch/agent/task/pickup/pickup_executor.py).

Some tasks called by the executor require an argument provided by the LLM. For example, the [_find](https://github.com/hello-robot/stretch_ai/blob/64c718773bad384599752ce6f52e6add9013b92d/src/stretch/agent/task/pickup/pickup_executor.py#L147) method takes a target_object as an argument, which enables it to find an object requested using natural language. The handover task does not take an argument, since it always looks for a person.

## Modify the LLM Prompt
 
With Stretch AI's [pick-and-place demo](https://github.com/hello-robot/stretch_ai/blob/main/docs/llm_agent.md), you can provide a natural language request to an LLM and the LLM will output text that specifies the tasks that the robot should perform. The LLM's text output is then parsed into a list of tuples with task identifiers and task arguments. This list then goes to the executor described in the previous section, which processes the list of tuples and executes the tasks. 
 
### Tell the LLM How to Use Your Task
 
To use your new task with an LLM, you need to provide natural language instructions to the LLM that tell it how to use your new task. These natural language instructions are included in a prompt provided to the LLM prior to receiving requests from a user. 

For the handover task, we edited the [LLM prompt text](https://github.com/hello-robot/stretch_ai/blob/64c718773bad384599752ce6f52e6add9013b92d/src/stretch/llms/prompts/pickup_prompt.py#L79) in [pickup_prompt.py](../src/stretch/llms/prompts/pickup_prompt.py), which is found in the [/src/stretch/llms/prompts](../src/stretch/llms/prompts) directory. 

First, we added a new command at [the beginning of the prompt](https://github.com/hello-robot/stretch_ai/blob/64c718773bad384599752ce6f52e6add9013b92d/src/stretch/llms/prompts/pickup_prompt.py#L21) that corresponds with the edits we made to the PickupExecutor.

```
When prompted, you will respond using these actions:
...
- place(location_name)  # location_name is the name of the receptacle to place object in
- say(text)  # say something to the user
- hand_over() # find a person, navigate to them, and extend the robot's hand toward them
...
- quit()  # end the conversation
```

Then, we added instructions on how to use this new action. For the handover task, we first added the following instruction followed by examples.

```
If you are asked to give the speaker an object and you have not already picked it up, you will first pick up the object and then hand it to the person.

Examples:

input: "Bring me the plastic toy."
output: 
say("I am picking up the plastic toy and handing it over to you.")
pickup(plastic toy)
hand_over()
end()

...
```

When using an LLM with Stretch AI, prior requests and actions are provided as context for a new request. So, **individual requests provided to the LLM are not treated independently.**

The second instruction we added attempts to address situations that depend on this context. 

```
If you are asked to give the speaker an object you are already holding, you should hand it over without picking it up again.

Examples: 

input: "Hand the item to me."
output: 
say("I am handing the object I am holding over to you.")
hand_over()
end()

...
```

### Convert LLM Output Text to Tuples

Finally, you need to edit the [parse_response](https://github.com/hello-robot/stretch_ai/blob/64c718773bad384599752ce6f52e6add9013b92d/src/stretch/llms/prompts/pickup_prompt.py#L221) method in [pickup_prompt.py](../src/stretch/llms/prompts/pickup_prompt.py) to convert the text output by the LLM into a tuple with a task identifier and task argument. 

### Test Your Prompt

The effectiveness of your prompt can depend on the specific LLM you are using. While performing prompt engineering, we tested the effectiveness of our prompts using the following command, which displays the list of task tuples output when given requests via speech. 

```
python -m stretch.app.chat --voice --llm qwen25 --prompt pickup
```

An example of running this command during handover task development follows.

```
(stretch_ai_0.1.15) cckemp@deep-linux:~/stretch_ai$ python -m stretch.app.chat --voice --llm qwen25 --prompt pickup
...
Press enter to speak or ctrl+c to exit.
Recording...
 32%|█████████████▌                             | 136/430 [00:03<00:06, 42.75it/s]
 ...
I heard:  Bring me the toy chicken
Response: [('pickup', 'toy chicken'), ('hand_over', '')]
...
Press enter to speak or ctrl+c to exit.
Recording...
 36%|███████████████▎                           | 153/430 [00:03<00:06, 42.54it/s]
...
I heard:  Pick up the toy cup.
Response: [('say', '"I am picking up the toy cup."'), ('pickup', 'toy cup')]
...
Press enter to speak or ctrl+c to exit.
Recording...
 25%|██████████▉                                | 109/430 [00:02<00:07, 42.53it/s]
Recording finished.
...
I heard:  Give it to me
Response: [('say', '"I am handing the toy cup to you."'), ('hand_over', '')]
...
```

Once you've tested your new task in isolation and tested your LLM prompt, you can try using your task in combination with other tasks. For example, we tested the new handover task with the existing pick and place tasks using the following command. 

```
python -m stretch.app.ai_pickup --use_llm --use_voice
```

