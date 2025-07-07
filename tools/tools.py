def calculator_tool(expression: str) -> str:
    """
    Evaluates a simple mathematical expression and returns the result as a string.
    Example: "2 + 2" => "4"
    """
    return str(eval(expression))


def weather_tool(location: str) -> str:
    """
    Returns a dummy weather response for a given location.
    Example: "Dhaka" => "The weather in Dhaka is sunny."
    """
    return f"The weather in {location} is sunny."


def fluency_tool(sentence: str) -> str:
    """
    Scores the fluency of a given Bangla sentence.
    Example: "আমি ভাত খাই।" => "Fluent"
    """
    return "Fluent"
