"""
Generator Agent — synthesizes a comprehensive, cited research report
from the retrieved and graded documents.
"""

from typing import List
from groq import Groq


GENERATOR_SYSTEM = """You are an expert research analyst. Your job is to synthesize 
information from multiple sources into a comprehensive, well-structured report.

Rules:
- Write in clear, professional prose
- Structure with markdown headers (##, ###)
- Cite sources inline using [Source: filename_or_url]
- If sources conflict, note the disagreement
- Separate facts from analysis
- End with a "## Key Takeaways" section
- Be thorough but concise — quality over length
"""

GENERATOR_USER = """Write a comprehensive research report answering this query:

QUERY: {query}

Use ONLY the information from the following sources. Cite each source inline.

--- SOURCES ---
{context}
--- END SOURCES ---

Write the report now:"""


def _build_context(documents: List[dict]) -> str:
    """Format documents into a context block for the LLM."""
    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.get("source", f"Source {i}")
        text   = doc.get("text", "").strip()
        doc_type = doc.get("type", "unknown")

        parts.append(
            f"[{i}] Source: {source} (type: {doc_type})\n"
            f"{text[:3000]}\n"
            f"{'─' * 40}"
        )
    return "\n\n".join(parts)


def generate_report(query: str, documents: List[dict], sources: List[str]) -> str:
    """
    Generate a comprehensive research report.

    Args:
        query:     The original research question
        documents: Graded, relevant documents
        sources:   List of source identifiers

    Returns:
        Markdown-formatted research report string
    """
    if not documents:
        return (
            f"## Research Report: {query}\n\n"
            "⚠️ No relevant documents were found — "
            "please add PDFs to the `sample_docs/` folder or check your search settings."
        )

    client  = Groq()
    context = _build_context(documents)

    response = client.chat.completions.create(
        model       = "llama-3.3-70b-versatile",
        max_tokens  = 4096,
        temperature = 0.2,
        messages    = [
            {"role": "system", "content": GENERATOR_SYSTEM},
            {"role": "user",   "content": GENERATOR_USER.format(
                query   = query,
                context = context
            )}
        ]
    )

    return response.choices[0].message.content
