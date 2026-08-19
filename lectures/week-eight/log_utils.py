import html


RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
BG_BLACK = "\033[40m"
BG_BLUE = "\033[44m"
RESET = "\033[0m"

COLOR_MAP = {
    BG_BLACK + RED: "#ff5555",
    BG_BLACK + GREEN: "#50fa7b",
    BG_BLACK + YELLOW: "#f1fa8c",
    BG_BLACK + BLUE: "#8be9fd",
    BG_BLACK + MAGENTA: "#ff79c6",
    BG_BLACK + CYAN: "#8be9fd",
    BG_BLACK + WHITE: "#f8f8f2",
    BG_BLUE + WHITE: "#ffb86c",
}


def reformat(message: str) -> str:
    """Escape a log line and convert the agents' ANSI colors to safe HTML spans."""
    formatted = html.escape(message)
    for ansi_code, color in COLOR_MAP.items():
        formatted = formatted.replace(ansi_code, f'<span style="color: {color}">')
    return formatted.replace(RESET, "</span>")
