from collections import Counter
from enum import Enum


class Color(Enum):
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    RED = "\033[91m"
    RESET = "\033[0m"


def colored(text: str, color: Color) -> str:
    """Return text colored with ANSI escape codes."""
    return f"{color.value}{text}{Color.RESET.value}"


def format_exec_plan(exec_plan: list[int]) -> str:
    """Format the execution plan for better readability."""
    if not exec_plan:
        return ""

    # Assume looped layers are those that appear more than once

    layer_counts = Counter(exec_plan)
    looped_layers = {layer for layer, count in layer_counts.items() if count > 1}

    def color_layer(layer):
        if layer in looped_layers:
            return colored(str(layer), Color.RED)
        else:
            return colored(str(layer), Color.GREEN)

    return " -> ".join(color_layer(layer) for layer in exec_plan)
