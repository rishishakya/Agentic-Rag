"""
Grader Agent — uses an LLM to score whether retrieved documents
are actually relevant to the query. Filters out noise.
"""

import json
import os
from typing import List

from groq import Groq


GRADER_PROMPT = """You are a document relevance grader.

Given a USER QUERY and a DOCUMENT CHUNK, decide if the document is relevant.

Respond with ONLY a JSON object:
{{"relevant": true}} or {{"relevant": false}}

Do not explain. Just output the JSON.

USER QUERY: {query}

DOCUMENT CHUNK:
{text}
"""


def grade_documents(query: str, documents: List[dict], threshold: float = 0.0) -> List[dict]:
    """
    Filter documents by LLM-judged relevance.

    Args:
        query:     The user's research query
        documents: List of doc dicts with 'text' key
        threshold: Minimum cosine score to even attempt grading (0 = grade all)

    Returns:
        Filtered list of relevant documents
    """
    if not documents:
        return []

    client  = Groq()
    relevant = []

    for doc in documents:
        # Quick score pre-filter (skip obviously bad retrievals)
        if doc.get("score", 1.0) < threshold:
            continue

        text = doc.get("text", "")[:2000]  # truncate to save tokens

        try:
            response = client.chat.completions.create(
                model      = "llama-3.3-70b-versatile",
                max_tokens = 20,
                temperature= 0,
                messages   = [
                    {
                        "role":    "user",
                        "content": GRADER_PROMPT.format(query=query, text=text)
                    }
                ]
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            raw = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)

            if result.get("relevant", False):
                relevant.append(doc)

        except (json.JSONDecodeError, Exception):
            # If grading fails, keep the document (fail open)
            relevant.append(doc)

    return relevant
