"""
Agentic RAG Orchestrator using LangGraph.

The agent follows this reasoning loop:
  1. Receive complex query
  2. Search local vector DB (ChromaDB)
  3. Grade retrieved docs — are they sufficient?
  4. If NOT sufficient → web search (Tavily/DuckDuckGo fallback)
  5. Re-grade combined context
  6. Generate final cited report
  7. Hallucination check → if fails, regenerate
"""

from typing import TypedDict, List, Annotated
import operator

from langgraph.graph import StateGraph, END

from agents.retriever   import retrieve_from_vectordb
from agents.grader      import grade_documents
from agents.web_search  import web_search
from agents.generator   import generate_report
from agents.hallucination_checker import check_hallucination
from utils.logger import log_step


# ── State Schema ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query:           str                          # original user query
    documents:       Annotated[List[dict], operator.add]  # accumulated docs
    web_searched:    bool                         # prevent infinite web loops
    generation:      str                          # final report text
    sources:         Annotated[List[str], operator.add]   # all source URLs/names
    iterations:      int                          # safety counter


# ── Node Functions ────────────────────────────────────────────────────────────

def node_retrieve(state: AgentState) -> dict:
    log_step("📚 Node: Retrieve", "Searching local vector database...")
    query = state["query"]
    docs  = retrieve_from_vectordb(query, k=5)
    sources = [d.get("source", "local-db") for d in docs]
    log_step("📚 Retrieve done", f"Found {len(docs)} local documents")
    return {"documents": docs, "sources": sources}


def node_grade_documents(state: AgentState) -> dict:
    log_step("🔍 Node: Grade", "Evaluating document relevance...")
    query    = state["query"]
    docs     = state["documents"]
    relevant = grade_documents(query, docs)
    log_step("🔍 Grade done", f"{len(relevant)}/{len(docs)} documents are relevant")
    return {"documents": relevant}


def node_web_search(state: AgentState) -> dict:
    log_step("🌐 Node: Web Search", "Fetching fresh data from the web...")
    query   = state["query"]
    results = web_search(query, num_results=5)
    sources = [r.get("url", "") for r in results]
    log_step("🌐 Web Search done", f"Got {len(results)} web results")
    return {
        "documents":    results,
        "web_searched": True,
        "sources":      sources
    }


def node_generate(state: AgentState) -> dict:
    log_step("✍️  Node: Generate", "Synthesizing final report...")
    report = generate_report(
        query     = state["query"],
        documents = state["documents"],
        sources   = list(set(state.get("sources", [])))
    )
    return {"generation": report, "iterations": state.get("iterations", 0) + 1}


def node_hallucination_check(state: AgentState) -> dict:
    """Passthrough node — routing happens in the edge below."""
    log_step("🧪 Node: Hallucination Check", "Verifying answer is grounded...")
    return {}


# ── Conditional Edges ─────────────────────────────────────────────────────────

def edge_should_web_search(state: AgentState) -> str:
    """After grading: enough docs? → generate. Otherwise → web search."""
    docs           = state.get("documents", [])
    already_searched = state.get("web_searched", False)

    if len(docs) >= 2:
        log_step("✅ Edge", "Sufficient docs — skipping web search")
        return "generate"

    if already_searched:
        log_step("⚠️  Edge", "Already web-searched, generating with what we have")
        return "generate"

    log_step("🔄 Edge", "Not enough docs — triggering web search")
    return "web_search"


def edge_hallucination_gate(state: AgentState) -> str:
    """After generation: is the answer grounded? → end. Otherwise → regenerate."""
    iterations = state.get("iterations", 0)
    if iterations >= 3:
        log_step("⚠️  Edge", "Max iterations reached — returning current answer")
        return "end"

    generation = state.get("generation", "")
    documents  = state.get("documents", [])
    passed     = check_hallucination(generation, documents)

    if passed:
        log_step("✅ Edge", "Hallucination check passed — done!")
        return "end"
    else:
        log_step("🔄 Edge", "Answer not grounded — regenerating...")
        return "regenerate"


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("retrieve",            node_retrieve)
    graph.add_node("grade_documents",     node_grade_documents)
    graph.add_node("web_search",          node_web_search)
    graph.add_node("generate",            node_generate)
    graph.add_node("hallucination_check", node_hallucination_check)

    # Entry point
    graph.set_entry_point("retrieve")

    # Edges
    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        edge_should_web_search,
        {"generate": "generate", "web_search": "web_search"}
    )

    graph.add_edge("web_search", "generate")
    graph.add_edge("generate",   "hallucination_check")

    graph.add_conditional_edges(
        "hallucination_check",
        edge_hallucination_gate,
        {"end": END, "regenerate": "generate"}
    )

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

def run_agentic_rag(query: str) -> dict:
    """
    Run the full Agentic RAG pipeline on a query.

    Returns:
        dict with 'report', 'sources', 'iterations'
    """
    app = build_graph()

    initial_state: AgentState = {
        "query":        query,
        "documents":    [],
        "web_searched": False,
        "generation":   "",
        "sources":      [],
        "iterations":   0,
    }

    log_step("🚀 Agentic RAG started", f"Query: {query}")
    final_state = app.invoke(initial_state)
    log_step("🏁 Pipeline complete", f"Iterations: {final_state.get('iterations', 0)}")

    return {
        "report":     final_state.get("generation", ""),
        "sources":    list(set(final_state.get("sources", []))),
        "iterations": final_state.get("iterations", 0),
        "doc_count":  len(final_state.get("documents", []))
    }
