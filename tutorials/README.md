# DeepAgent SDK – Tutorials

This directory contains step-by-step tutorials showing how to use the
**DeepAgent** SDK.

## Prerequisites

```bash
pip install deepagent openai
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

## Tutorial files

| File | What you will learn |
|---|---|
| `01_getting_started.py` | Install & run your first agent |
| `02_built_in_tools.py` | Using the built-in calculator, word counter, and more |
| `03_custom_tools.py` | Defining your own tools with the `@tool` decorator |
| `04_memory.py` | Conversation memory – full history vs sliding window |
| `05_multi_turn_chat.py` | Multi-turn conversations with persistent memory |

## Running the tutorials

Each tutorial is a self-contained Python script.  Run them from the repo root:

```bash
python tutorials/01_getting_started.py
python tutorials/02_built_in_tools.py
# etc.
```

> **Note** – tutorials 01–05 make real LLM API calls and will consume API
> credits.  Make sure `OPENAI_API_KEY` is set before running them.
