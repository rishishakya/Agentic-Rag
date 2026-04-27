"""Colored terminal logger."""

import datetime

COLORS = {
    "cyan":   "\033[96m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}

def log_step(title: str, detail: str = "", color: str = "cyan") -> None:
    ts    = datetime.datetime.now().strftime("%H:%M:%S")
    c     = COLORS.get(color, COLORS["cyan"])
    reset = COLORS["reset"]
    bold  = COLORS["bold"]
    print(f"{bold}{c}[{ts}] {title}{reset}", flush=True)
    if detail:
        print(f"           {detail[:200]}", flush=True)
