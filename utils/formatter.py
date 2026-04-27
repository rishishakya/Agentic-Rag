"""Format and save research reports."""

import datetime
from typing import List

DIVIDER = "═" * 70

def format_final_report(query: str, report: str, sources: List[str],
                         iterations: int, doc_count: int) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    unique_sources = list(set(s for s in sources if s))

    lines = [
        "",
        DIVIDER,
        "  🔬  AGENTIC RAG — RESEARCH REPORT",
        DIVIDER,
        f"  Query      : {query}",
        f"  Generated  : {ts}",
        f"  Documents  : {doc_count} chunks used",
        f"  Iterations : {iterations} reasoning loop(s)",
        DIVIDER,
        "",
        report,
        "",
    ]

    if unique_sources:
        lines += [DIVIDER, "  📚  All Sources:"]
        for i, src in enumerate(unique_sources, 1):
            lines.append(f"  [{i}] {src}")

    lines += [DIVIDER, ""]
    return "\n".join(lines)


def save_report(report: str, filename: str = "") -> str:
    if not filename:
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{ts}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    return filename
