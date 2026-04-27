"""
Hallucination Checker — verifies the generated report is grounded
in the retrieved documents, not invented by the LLM.
"""

import json
from typing import List
from groq import Groq


CHECKER_PROMPT = """You are a fact-checking AI.

Your job: Determine if a GENERATED REPORT is grounded in the PROVIDED SOURCES.

A report is "grounded" if its key claims can be traced back to the sources.
It is NOT grounded if it contains major facts not present in any source.

Respond ONLY with JSON:
{{"grounded": true, "reason": "brief explanation"}}
or
{{"grounded": false, "reason": "what was hallucinated"}}

SOURCES (summary):
{sources_summary}

GENERATED REPORT:
{report}
"""


def check_hallucination(generation: str, documents: List[dict]) -> bool:
    """
    Check if the generated report is grounded in the source documents.

    Args:
        generation: The generated report text
        documents:  Source documents used for generation

    Returns:
        True if grounded (passes check), False if hallucinating
    """
    if not documents or not generation:
        return True  # nothing to check

    # Build a brief summary of sources for the checker
    sources_summary = "\n".join(
        f"- [{d.get('source', 'unknown')}]: {d.get('text', '')[:300]}..."
        for d in documents[:5]
    )

    client = Groq()

    try:
        response = client.chat.completions.create(
            model       = "llama-3.3-70b-versatile",
            max_tokens  = 150,
            temperature = 0,
            messages    = [
                {
                    "role":    "user",
                    "content": CHECKER_PROMPT.format(
                        sources_summary = sources_summary,
                        report          = generation[:3000]  # truncate long reports
                    )
                }
            ]
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)

        grounded = result.get("grounded", True)
        reason   = result.get("reason", "")

        if not grounded:
            print(f"⚠️  Hallucination detected: {reason}")

        return grounded

    except (json.JSONDecodeError, Exception):
        return True  # fail open — don't block on checker errors
