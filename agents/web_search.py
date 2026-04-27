"""
Web Search Agent — fetches fresh data when local docs are insufficient.
Uses Tavily API if key is set, falls back to DuckDuckGo (free, no key).
"""

import os
import time
from typing import List


def _search_tavily(query: str, num_results: int) -> List[dict]:
    """Search using Tavily API (best quality, requires API key)."""
    from tavily import TavilyClient

    client  = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    results = client.search(
        query          = query,
        max_results    = num_results,
        search_depth   = "advanced",
        include_answer = False
    )

    docs = []
    for r in results.get("results", []):
        docs.append({
            "text":    r.get("content", ""),
            "source":  r.get("url", ""),
            "title":   r.get("title", ""),
            "type":    "web"
        })
    return docs


def _search_duckduckgo(query: str, num_results: int) -> List[dict]:
    """Fallback: DuckDuckGo search (free, no key needed)."""
    from duckduckgo_search import DDGS

    docs = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=num_results):
            docs.append({
                "text":   r.get("body", ""),
                "source": r.get("href", ""),
                "title":  r.get("title", ""),
                "type":   "web"
            })
            time.sleep(0.1)
    return docs


def web_search(query: str, num_results: int = 5) -> List[dict]:
    """
    Search the web for fresh information.

    Tries Tavily first (if TAVILY_API_KEY is set), then falls back to DuckDuckGo.

    Args:
        query:       Search query
        num_results: Number of results to retrieve

    Returns:
        List of doc dicts with 'text', 'source', 'title', 'type'
    """
    # Try Tavily first (higher quality, supports RAG-optimized results)
    if os.environ.get("TAVILY_API_KEY"):
        try:
            results = _search_tavily(query, num_results)
            if results:
                return results
        except ImportError:
            pass  # tavily not installed
        except Exception as e:
            print(f"⚠️  Tavily failed: {e} — falling back to DuckDuckGo")

    # Fallback: DuckDuckGo (always free)
    try:
        return _search_duckduckgo(query, num_results)
    except Exception as e:
        print(f"⚠️  DuckDuckGo failed: {e}")
        return []
