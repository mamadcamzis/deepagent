"""
Tutorial 04 – Memory Management
==================================

DeepAgent provides two memory strategies:

ConversationMemory (default)
    Keeps the *entire* conversation history.  Simple but the context window
    grows without bound for long sessions.

SlidingWindowMemory
    Keeps only the *k* most recent messages.  Ideal for long-running agents
    where you want to limit token usage.

Prerequisites
-------------
    pip install deepagent openai
    export OPENAI_API_KEY="sk-..."
"""

from deepagent import Agent, ConversationMemory, DeepAgentConfig, SlidingWindowMemory

config = DeepAgentConfig(model="gpt-4o-mini", verbose=False)

# ---------------------------------------------------------------------------
# Part A – ConversationMemory (default)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Part A – Full ConversationMemory")
print("=" * 60)

memory = ConversationMemory(system_prompt="You are a helpful assistant.")
agent = Agent(config=config, memory=memory)

agent.run("My favourite colour is blue.")
agent.run("I love Italian food.")
response = agent.run("What do you know about me so far?")

print(f"\nAgent says: {response.output}")
print(f"Messages in memory: {len(memory)}")

# ---------------------------------------------------------------------------
# Part B – SlidingWindowMemory
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Part B – SlidingWindowMemory (window=4)")
print("=" * 60)

# Only the 4 most recent non-system messages are kept.
sliding_memory = SlidingWindowMemory(window_size=4)
agent2 = Agent(config=config, memory=sliding_memory)

for msg in [
    "Fact 1: The sky is blue.",
    "Fact 2: Water is wet.",
    "Fact 3: Fire is hot.",
    "Fact 4: Ice is cold.",
    "Fact 5: Grass is green.",  # This should push out Fact 1
]:
    agent2.run(msg)

# Only the last 4 non-system messages should remain
print(f"\nMessages in sliding window: {len(sliding_memory)}")
print("Recent messages:")
for m in sliding_memory.get_messages():
    role = m.role.upper()
    preview = m.content[:80].replace("\n", " ")
    print(f"  [{role}] {preview}")

response2 = agent2.run("What was the very first fact I told you?")
print(f"\nAgent says: {response2.output}")
print("(The agent should NOT remember Fact 1 – it was evicted by the sliding window.)")

# ---------------------------------------------------------------------------
# Part C – Manually inspecting and clearing memory
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Part C – Inspecting and clearing memory")
print("=" * 60)

memory3 = ConversationMemory(system_prompt="You are a math tutor.")
agent3 = Agent(config=config, memory=memory3)

agent3.run("What is 5 + 3?")
agent3.run("And what is that result times 4?")

print(f"Messages before reset: {len(memory3)}")
for m in memory3.get_messages():
    print(f"  [{m.role}] {m.content[:60]}")

# Reset keeps the system prompt but clears the rest
agent3.reset()
print(f"\nMessages after reset (keep_system=True): {len(memory3)}")
for m in memory3.get_messages():
    print(f"  [{m.role}] {m.content[:60]}")
