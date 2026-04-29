# DeepAgent

A lightweight Python SDK for building **LLM-powered agents** with tools and memory.

DeepAgent follows the **ReAct** (Reasoning + Acting) pattern: it lets an LLM
decide which tools to call, interprets the results, and keeps reasoning until
it can produce a confident final answer.

---

## Features

- 🤖 **ReAct agent loop** – iterative reasoning + tool calling
- 🔧 **Built-in tools** – calculator, word counter, JSON formatter, and more
- 🛠️ **Custom tools** – register any Python function with a single decorator
- 🧠 **Flexible memory** – full conversation history or a sliding window
- ⚙️ **Simple configuration** – one dataclass to control model, temperature, verbosity
- 📦 **Minimal dependencies** – only requires `openai`

---

## Installation

```bash
pip install deepagent
```

Or install directly from source:

```bash
git clone https://github.com/mamadcamzis/deepagent.git
cd deepagent
pip install -e .
```

---

## Quick Start

```python
import os
from deepagent import Agent, DeepAgentConfig

# 1. Configure
config = DeepAgentConfig(
    model="gpt-4o-mini",
    max_iterations=5,
    verbose=True,
)

# 2. Create the agent (reads OPENAI_API_KEY from environment)
agent = Agent(config=config)

# 3. Run!
response = agent.run("What is the square root of 144 multiplied by 7?")
print(response.output)
# → 84.0
```

---

## Core concepts

### Configuration

`DeepAgentConfig` is the single place to set model, temperature, and behaviour:

```python
from deepagent import DeepAgentConfig

config = DeepAgentConfig(
    model="gpt-4o-mini",     # any OpenAI chat model
    api_key="sk-...",        # or set OPENAI_API_KEY env var
    max_iterations=10,       # max reasoning steps
    temperature=0.0,         # 0 = deterministic
    verbose=True,            # print the reasoning trace
)

# Or build from environment variables:
config = DeepAgentConfig.from_env()
```

### Agent

```python
from deepagent import Agent

agent = Agent(config=config)

response = agent.run("Summarise the key points of quantum computing.")

print(response.output)          # final answer string
print(response.iterations)      # how many LLM calls were made
print(response.steps)           # list of intermediate steps

agent.reset()                   # clear memory, keep system prompt
```

### Built-in Tools

| Tool | What it does |
|---|---|
| `calculator` | Evaluates safe arithmetic expressions (`sqrt`, `**`, etc.) |
| `word_count` | Counts words in a string |
| `json_formatter` | Pretty-prints a JSON string |
| `text_reverser` | Reverses a string |
| `list_creator` | Turns comma-separated items into a numbered list |

```python
from deepagent import list_tools

for t in list_tools():
    print(t.name, "–", t.description[:60])
```

### Custom Tools

Use the `@tool` decorator to register any Python function as a tool:

```python
from deepagent import Agent, DeepAgentConfig, tool

@tool(
    name="shout",
    description="Convert text to UPPER CASE. Input: any string.",
)
def shout(text: str) -> str:
    return text.upper()

agent = Agent(config=DeepAgentConfig())
response = agent.run("Please shout the word 'hello'.")
print(response.output)  # → HELLO
```

You can also create a `Tool` manually (without the decorator) and pass it to
the `Agent` via `tools=[...]`:

```python
from deepagent import Agent, Tool

my_tool = Tool(
    name="greet",
    description="Greet a person by name. Input: a person's name.",
    func=lambda name: f"Hello, {name}!",
)

agent = Agent(config=config, tools=[my_tool])
```

### Memory

```python
from deepagent import Agent, ConversationMemory, SlidingWindowMemory

# Full history (default)
memory = ConversationMemory(system_prompt="You are a helpful assistant.")

# Keep only the last 10 messages
memory = SlidingWindowMemory(window_size=10)

agent = Agent(config=config, memory=memory)
```

---

## Tutorials

The [`tutorials/`](tutorials/) directory contains five hands-on examples:

| Tutorial | Topic |
|---|---|
| [`01_getting_started.py`](tutorials/01_getting_started.py) | Your first agent |
| [`02_built_in_tools.py`](tutorials/02_built_in_tools.py) | Using built-in tools |
| [`03_custom_tools.py`](tutorials/03_custom_tools.py) | Creating custom tools |
| [`04_memory.py`](tutorials/04_memory.py) | Memory strategies |
| [`05_multi_turn_chat.py`](tutorials/05_multi_turn_chat.py) | Interactive chatbot |

```bash
export OPENAI_API_KEY="sk-..."
python tutorials/01_getting_started.py
```

---

## Project layout

```
deepagent/
├── deepagent/
│   ├── __init__.py      # public API
│   ├── agent.py         # Agent class (ReAct loop)
│   ├── config.py        # DeepAgentConfig
│   ├── memory.py        # ConversationMemory, SlidingWindowMemory
│   └── tools.py         # Tool class, @tool decorator, built-in tools
├── tutorials/
│   ├── 01_getting_started.py
│   ├── 02_built_in_tools.py
│   ├── 03_custom_tools.py
│   ├── 04_memory.py
│   └── 05_multi_turn_chat.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Requirements

- Python ≥ 3.10
- `openai >= 1.0.0`

---

## License

MIT