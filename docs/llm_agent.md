# The Stretch AI Agent

Stretch AI contains the tools to talk to your robot and have it perform tasks like exploration, mapping, and pick-and-place. In this document, we'll walk through what the AI agent is, how it works, and how to test out different components of the LLM.

<div class="embed-container">
    <iframe width="640" height="390" 
    src="https://youtu.be/oO9qRkiuiAQ"
    frameborder="0" allowfullscreen></iframe>
</div>

Above: example of the AI agent being used with the voice command and the open-source [Qwen2.5 LLM](https://huggingface.co/Qwen). The specific commands used in the video are:
```
python -m stretch.app.ai_pickup --use_llm --use_voice
```

## What is the AI Agent?

When you run the `ai_pickup` command, it will create a [Pickup Executor](src/stretch/agent/task/pickup/pickup_executor.py) object, which parses instructions from either an LLM or from a handful of templates and uses them to create a reactive task plan for the robot to execute. When you use this with the `--use_llm` flag -- which is recommended -- it will instantiate one of a number of LLM clients to generate the instructions. The LLM clients are defined in the [llm_agent.py](src/stretch/agent/llm/__init__.py) file and are:
  - `qwen25` and variants: the Qwen2.5 model from Tencent; a permissively-licensed model. The default is `qwen25-3B-Instruct`.
  - `openai`: the OpenAI GPT-4o-mini model; a proprietary model accessed through the OpenAI API.
  - `gemma2b`: the Gemma2b model from Google, accessed via Hugging Face's model hub.

We recommend `qwen25-3B-Instruct` or `gemma2b` if running locally on a powerful machine (e.g. a computer with an NVIDIA 4090 or similar), and `openai` if you have access to the OpenAI API.

For example if you want to test with Gemma 2b, you can run:
```bash
python -m stretch.app.ai_pickup --use_llm --llm gemma2b
```

### Using OpenAI Models with Stretch AI

To use an OpenAI model, first create an OpenAI API KEY by following the [OpenAI quickstart instructions](https://platform.openai.com/docs/quickstart). Then, set the `OPENAI_API_KEY` environment variable to your API key. You can do this by adding the following line to your `~/.bashrc` or `~/.bash_profile` file:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Then, restart your terminal or run `source ~/.bashrc` or `source ~/.bash_profile` to apply the changes.

You can specify that you want to use the OpenAI model by passing the `--llm openai` flag to the `ai_pickup` command:
```bash
python -m stretch.app.ai_pickup --use_llm --llm openai
```


