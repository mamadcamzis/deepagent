"""
Tutorial 02 – Using Built-in Tools
===================================

DeepAgent ships with several ready-to-use tools.  This tutorial exercises
each of them so you can see how the agent picks and uses the right tool for
each query.

Built-in tools
--------------
- calculator    – safe arithmetic expression evaluator
- word_count    – counts words in a piece of text
- json_formatter – pretty-prints a JSON string
- text_reverser – reverses a string
- list_creator  – turns comma-separated values into a numbered list

Prerequisites
-------------
    pip install deepagent openai
    export OPENAI_API_KEY="sk-..."
"""

from deepagent import Agent, DeepAgentConfig, list_tools

# ---------------------------------------------------------------------------
# Print all registered tools
# ---------------------------------------------------------------------------
print("=" * 60)
print("Registered tools")
print("=" * 60)
for t in list_tools():
    print(f"  • {t.name:20s} {t.description[:60]}…")

# ---------------------------------------------------------------------------
# Create an agent
# ---------------------------------------------------------------------------
config = DeepAgentConfig(model="gpt-4o-mini", verbose=True)
agent = Agent(config=config)

# ---------------------------------------------------------------------------
# Example 1 – Calculator
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Example 1 – Calculator")
print("=" * 60)
response = agent.run("Calculate: (sqrt(144) + 3**2) * 2")
print(f"\nAnswer: {response.output}")

agent.reset()

# ---------------------------------------------------------------------------
# Example 2 – Word counter
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Example 2 – Word counter")
print("=" * 60)
response = agent.run(
    "How many words are in the following text?\n\n"
    '"The quick brown fox jumps over the lazy dog."'
)
print(f"\nAnswer: {response.output}")

agent.reset()

# ---------------------------------------------------------------------------
# Example 3 – JSON formatter
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Example 3 – JSON formatter")
print("=" * 60)
ugly_json = '{"name":"Alice","age":30,"skills":["Python","Go","Rust"]}'
response = agent.run(f"Please pretty-print this JSON for me:\n{ugly_json}")
print(f"\nAnswer:\n{response.output}")

agent.reset()

# ---------------------------------------------------------------------------
# Example 4 – List creator
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Example 4 – List creator")
print("=" * 60)
response = agent.run(
    "Create a numbered list from the following items: "
    "apples, bananas, cherries, dates, elderberries"
)
print(f"\nAnswer:\n{response.output}")
