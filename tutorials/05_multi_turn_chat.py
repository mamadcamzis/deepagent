"""
Tutorial 05 – Multi-turn Conversations
=========================================

In this tutorial we build a small interactive chatbot that demonstrates:

- Persistent memory across multiple turns
- The agent remembering context from earlier in the conversation
- Graceful exit

Prerequisites
-------------
    pip install deepagent openai
    export OPENAI_API_KEY="sk-..."

Usage
-----
    python tutorials/05_multi_turn_chat.py

Type a message and press Enter.  Type 'quit' or 'exit' to stop.
"""

from deepagent import Agent, DeepAgentConfig, SlidingWindowMemory

# ---------------------------------------------------------------------------
# Set up agent with a sliding window so it doesn't run out of context
# ---------------------------------------------------------------------------
config = DeepAgentConfig(
    model="gpt-4o-mini",
    max_iterations=5,
    verbose=False,  # set True to see the reasoning trace
)

memory = SlidingWindowMemory(
    window_size=20,  # keep the last 20 messages
    system_prompt=(
        "You are DeepBot, a friendly and knowledgeable AI assistant. "
        "You have access to various tools and will use them when helpful. "
        "Be concise but thorough in your responses."
    ),
)

agent = Agent(config=config, memory=memory)

# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------
print("=" * 60)
print("  DeepAgent Multi-turn Chat Demo")
print("=" * 60)
print("Type your message and press Enter. Type 'quit' to exit.\n")

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break

    if not user_input:
        continue

    if user_input.lower() in {"quit", "exit", "bye"}:
        print("DeepBot: Goodbye! It was nice chatting with you.")
        break

    if user_input.lower() == "/history":
        # Hidden command: show memory contents
        print(f"\n[Memory – {len(memory)} messages]")
        for m in memory.get_messages():
            print(f"  [{m.role.upper():10s}] {m.content[:80]}")
        print()
        continue

    if user_input.lower() == "/reset":
        agent.reset()
        print("DeepBot: Memory cleared. Starting fresh!\n")
        continue

    response = agent.run(user_input)
    print(f"DeepBot: {response.output}\n")
