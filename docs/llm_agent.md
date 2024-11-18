# The Stretch AI Agent

Stretch AI contains the tools to talk to your robot and have it perform tasks like exploration, mapping, and pick-and-place. In this document, we'll walk through what the AI agent is, how it works, and how to test out different components of the LLM.

[![Example of Stretch AI using Voice Control](https://img.youtube.com/vi/oO9qRkiuiAQ/0.jpg)](https://www.youtube.com/watch?v=oO9qRkiuiAQ)

[Above](https://www.youtube.com/watch?v=oO9qRkiuiAQ): example of the AI agent being used with the voice command and the open-source [Qwen2.5 LLM](https://huggingface.co/Qwen). The specific commands used in the video are:
```
python -m stretch.app.ai_pickup --use_llm --use_voice
```

## Running the AI Agent

### What is the AI Agent?

When you run the `ai_pickup` command, it will create a [Pickup Executor](src/stretch/agent/task/pickup/pickup_executor.py) object, which parses instructions from either an LLM or from a handful of templates and uses them to create a reactive task plan for the robot to execute. When you use this with the `--use_llm` flag -- which is recommended -- it will instantiate one of a number of LLM clients to generate the instructions. The LLM clients are defined in the [llm_agent.py](src/stretch/agent/llm/__init__.py) file and are:
  - `qwen25` and variants: the Qwen2.5 model from Tencent; a permissively-licensed model. The default is `qwen25-3B-Instruct`.
  - `openai`: the OpenAI GPT-4o-mini model; a proprietary model accessed through the OpenAI API.
  - `gemma2b`: the Gemma2b model from Google, accessed via Hugging Face's model hub.

We recommend `qwen25-3B-Instruct` or `gemma2b` if running locally on a powerful machine (e.g. a computer with an NVIDIA 4090 or similar), and `openai` if you have access to the OpenAI API.

For example if you want to test with Gemma 2b, you can run:
```bash
python -m stretch.app.ai_pickup --use_llm --llm gemma2b
```

#### Using OpenAI Models with Stretch AI

To use an OpenAI model, first create an OpenAI API KEY by following the [OpenAI quickstart instructions](https://platform.openai.com/docs/quickstart). Then, set the `OPENAI_API_KEY` environment variable to your API key. You can do this by adding the following line to your `~/.bashrc` or `~/.bash_profile` file:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Then, restart your terminal or run `source ~/.bashrc` or `source ~/.bash_profile` to apply the changes.

You can specify that you want to use the OpenAI model by passing the `--llm openai` flag to the `ai_pickup` command:
```bash
python -m stretch.app.ai_pickup --use_llm --llm openai
```

### Testing the LLM Agent

You can use an LLM to provide free-form text input to the pick and place demo with the `--use_llm` command line argument.

Running the following command will first download an open LLM model. Currently, the default model is [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct). Running this command downloads ~10GB of data. Using an ethernet cable instead of Wifi is recommended.

```bash
python -m stretch.app.ai_pickup --use_llm
```

Once it's ready, you should see the prompt `You:` after which you can write your text request. Pressing the `Enter` key on your keyboard will provide your request to the robot.

For example, the following requests have been successful for other users.

```
You: pick up the toy chicken and put it in the white laundry basket
```

```
You: Find a toy chicken
```

Currently, the prompt used by the LLM encourages the robot to both pick and place, so you may find that a primitive request results in the full demonstration task.

You can find the prompt used by the LLM at the following location. When running your Docker image in the development mode or running *stretch-ai* from source, you can modify this file to see how it changes the robot's behavior.

[./src/stretch/llms/prompts/pickup_prompt.py](./src/stretch/llms/prompts/pickup_prompt.py)

## Agent Architecture

The entry point into the LLM Agent is the [ai_pickup.py](src/stretch/app/ai_pickup.py) file. This file creates an instance of the [PickupExecutor](src/stretch/agent/task/pickup/pickup_executor.py) class, which is responsible for parsing the instructions from the LLM and creating a task plan for the robot to execute.

In addition, if you ruin it wil the `--use_llm` flag, it creates a chat wrapper:
```python
 if use_llm:
        llm_client = get_llm_client(llm, prompt=prompt)
        chat_wrapper = LLMChatWrapper(llm_client, prompt=prompt, voice=use_voice)
```

This will create an LLM client (for example, the [OpenAI client](src/stretch/llms/openai_client.py)), and provide it a [prompt](src/stretch/llms/prompts). The prompt used in the LLM Agent demo is the [pickup_prompt.py](src/stretch/llms/prompts/pickup_prompt.py).

Take a look at how the prompt starts out:
> You are a friendly, helpful robot named Stretch. You are always helpful, and answer questions concisely. You will never harm a human or suggest harm.
> 
> When prompted, you will respond using these actions:
> - pickup(object_name)  # object_name is the name of the object to pick up
> - explore(int)  # explore the environment for a certain number of steps
> - place(location_name)  # location_name is the name of the receptacle to place object in
> - say(text)  # say something to the user
> - wave()  # wave at a person
> - nod_head() # nod your head
> - shake_head() # shake your head
> - avert_gaze() # avert your gaze
> - find(object_name)  # find the object or location by exploring
> - go_home()  # navigate back to where you started
> - quit()  # end the conversation

This lists the different functions that the LLM agent can use to interact with the world.




