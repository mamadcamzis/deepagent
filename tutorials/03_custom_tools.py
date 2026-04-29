"""
Tutorial 03 – Creating Custom Tools
=====================================

The `@tool` decorator lets you expose any Python function to the agent as
a callable capability.  This tutorial shows three practical custom tools:

1. ``unit_converter``   – converts between common units of measurement
2. ``string_stats``     – returns character/word/line stats for text
3. ``temperature_check``– decides whether a Celsius temperature is hot/cold

Prerequisites
-------------
    pip install deepagent openai
    export OPENAI_API_KEY="sk-..."
"""

from deepagent import Agent, DeepAgentConfig, Tool, tool

# ---------------------------------------------------------------------------
# Define custom tools using the @tool decorator
# ---------------------------------------------------------------------------

@tool(
    name="unit_converter",
    description=(
        "Convert a value between common units. "
        "Input format: '<value> <from_unit> to <to_unit>'. "
        "Supported: km/miles, kg/lbs, celsius/fahrenheit, liters/gallons."
    ),
)
def unit_converter(query: str) -> str:
    """Convert between common measurement units."""
    query = query.lower().strip()

    conversions = {
        ("km", "miles"):      lambda v: v * 0.621371,
        ("miles", "km"):      lambda v: v * 1.60934,
        ("kg", "lbs"):        lambda v: v * 2.20462,
        ("lbs", "kg"):        lambda v: v * 0.453592,
        ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
        ("liters", "gallons"): lambda v: v * 0.264172,
        ("gallons", "liters"): lambda v: v * 3.78541,
    }

    # Parse "10 km to miles"
    import re
    match = re.match(r"([\d.]+)\s+(\w+)\s+to\s+(\w+)", query)
    if not match:
        return "Invalid format. Use: '<value> <from_unit> to <to_unit>'"

    value = float(match.group(1))
    from_unit = match.group(2)
    to_unit = match.group(3)

    key = (from_unit, to_unit)
    if key not in conversions:
        return f"Unsupported conversion: {from_unit} → {to_unit}"

    result = conversions[key](value)
    return f"{value} {from_unit} = {result:.4f} {to_unit}"


@tool(
    name="string_stats",
    description=(
        "Return character count, word count, and line count for a text. "
        "Input: any string of text."
    ),
)
def string_stats(text: str) -> str:
    """Return basic text statistics."""
    chars = len(text)
    words = len(text.split())
    lines = len(text.splitlines()) or 1
    return f"Characters: {chars}, Words: {words}, Lines: {lines}"


@tool(
    name="temperature_check",
    description=(
        "Classify a Celsius temperature as 'freezing', 'cold', 'mild', "
        "'warm', or 'hot'. Input: a number (temperature in Celsius)."
    ),
)
def temperature_check(temp_str: str) -> str:
    """Classify a Celsius temperature."""
    temp = float(temp_str.strip())
    if temp < 0:
        label = "freezing"
    elif temp < 10:
        label = "cold"
    elif temp < 20:
        label = "mild"
    elif temp < 30:
        label = "warm"
    else:
        label = "hot"
    return f"{temp}°C is {label}"


# ---------------------------------------------------------------------------
# You can also build a Tool manually (without the decorator)
# ---------------------------------------------------------------------------

def _palindrome_check(word: str) -> str:
    w = word.strip().lower()
    return f"'{w}' is {'a palindrome' if w == w[::-1] else 'not a palindrome'}"

palindrome_tool = Tool(
    name="palindrome_check",
    description=(
        "Check whether a word or phrase is a palindrome. "
        "Input: a single word or short phrase (spaces and case ignored)."
    ),
    func=_palindrome_check,
)

# ---------------------------------------------------------------------------
# Create an agent with the custom tools
# ---------------------------------------------------------------------------
# The globally registered tools (via @tool) are picked up automatically.
# The manually created tool must be passed explicitly via tools=[...].

config = DeepAgentConfig(model="gpt-4o-mini", verbose=True)
agent = Agent(config=config, tools=[palindrome_tool])

# ---------------------------------------------------------------------------
# Example 1 – Unit converter
# ---------------------------------------------------------------------------
print("=" * 60)
print("Example 1 – Unit converter")
print("=" * 60)
response = agent.run("Convert 100 km to miles.")
print(f"\nAnswer: {response.output}")

agent.reset()

# ---------------------------------------------------------------------------
# Example 2 – String stats
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Example 2 – String stats")
print("=" * 60)
response = agent.run(
    "Give me stats for this text:\n"
    "DeepAgent makes it easy to build powerful AI workflows."
)
print(f"\nAnswer: {response.output}")

agent.reset()

# ---------------------------------------------------------------------------
# Example 3 – Palindrome check
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Example 3 – Palindrome check")
print("=" * 60)
response = agent.run("Is 'racecar' a palindrome?")
print(f"\nAnswer: {response.output}")

agent.reset()

# ---------------------------------------------------------------------------
# Example 4 – Temperature classification
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Example 4 – Temperature classification")
print("=" * 60)
response = agent.run("Is 37 degrees Celsius hot?")
print(f"\nAnswer: {response.output}")
