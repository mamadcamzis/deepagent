"""
Tutorial 01 – Getting Started with DeepAgent
============================================

This tutorial shows the minimum code needed to create an agent and ask it a
question.

Prerequisites
-------------
    pip install deepagent openai
    export OPENAI_API_KEY="sk-..."
"""

# ---------------------------------------------------------------------------
# Step 1 – import the SDK
# ---------------------------------------------------------------------------
from deepagent import Agent, DeepAgentConfig

# ---------------------------------------------------------------------------
# Step 2 – configure the agent
# ---------------------------------------------------------------------------
# DeepAgentConfig controls which model is used, how many iterations the agent
# is allowed to take, and whether debug output is printed.

config = DeepAgentConfig(
    model="gpt-4o-mini",   # any OpenAI chat model works here
    max_iterations=5,      # stop after 5 reasoning steps at most
    verbose=True,          # print each step to stdout
)

# ---------------------------------------------------------------------------
# Step 3 – create the agent
# ---------------------------------------------------------------------------
# When no api_key= is passed, the SDK reads OPENAI_API_KEY from the
# environment automatically.

agent = Agent(config=config)

# ---------------------------------------------------------------------------
# Step 4 – ask a question
# ---------------------------------------------------------------------------
print("=" * 60)
print("Example 1 – simple factual question")
print("=" * 60)

response = agent.run("What is the capital of France?")

print(f"\nAnswer : {response.output}")
print(f"Steps  : {response.iterations} LLM call(s)")

# ---------------------------------------------------------------------------
# Step 5 – reset and ask another question
# ---------------------------------------------------------------------------
# reset() clears the conversation history so the agent starts fresh.
agent.reset()

print("\n" + "=" * 60)
print("Example 2 – arithmetic question (uses the calculator tool)")
print("=" * 60)

response = agent.run("What is 144 divided by 12, then multiplied by 7?")
print(f"\nAnswer : {response.output}")
print(f"Steps  : {response.iterations} LLM call(s)")

# ---------------------------------------------------------------------------
# Step 6 – inspect the reasoning trace
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Reasoning trace")
print("=" * 60)
for step in response.steps:
    print(f"  [{step['iteration']}] action={step['action']!r}  input={step['input']!r}")
    if "output" in step:
        print(f"       output={step['output']!r}")
