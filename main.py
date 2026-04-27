#!/usr/bin/env python3
"""
Agentic RAG — CLI entry point.

Usage:
    python main.py                                    # interactive
    python main.py "Analyze carbon tax impact on EVs" # single query
    python main.py "..." --save                       # save to .md file
    python main.py "..." --quiet                      # no step logs
"""

import argparse
import sys
import os


def check_env():
    if not os.environ.get("GROQ_API_KEY"):
        print("❌  GROQ_API_KEY not set.")
        print()
        print("    1. Get a FREE key at: https://console.groq.com/keys")
        print("    2. PowerShell: $env:GROQ_API_KEY=\"your_key\"")
        print("       CMD:        set GROQ_API_KEY=your_key")
        sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(
        description="🔬 Agentic RAG — autonomous multi-source research system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Analyze the 2026 impact of carbon taxes on EV stocks"
  python main.py "What are the risks of investing in renewable energy ETFs?" --save
  python main.py --quiet "Compare LangGraph vs CrewAI for agentic systems"

First time? Ingest some documents first:
  python ingest.py
        """
    )
    p.add_argument("query",   nargs="?", help="Research query (omit for interactive mode)")
    p.add_argument("--save",  "-s", action="store_true", help="Save report to .md file")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress step logs")
    return p.parse_args()


def run(query: str, save: bool, quiet: bool):
    import logging
    if quiet:
        logging.disable(logging.CRITICAL)

    from graph import run_agentic_rag
    from utils.formatter import format_final_report, save_report

    result = run_agentic_rag(query)

    final = format_final_report(
        query      = query,
        report     = result["report"],
        sources    = result["sources"],
        iterations = result["iterations"],
        doc_count  = result["doc_count"]
    )

    print(final)

    if save:
        path = save_report(final)
        print(f"💾  Report saved to: {path}")


def interactive(args):
    print("\n🔬  Agentic RAG Research System")
    print("    Powered by LangGraph + ChromaDB + Groq (Free!)")
    print("    Type 'quit' to exit\n")

    while True:
        try:
            query = input("🔭 Research query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  Goodbye!")
            break

        if query.lower() in ("quit", "exit", "q"):
            print("👋  Goodbye!")
            break

        if not query:
            continue

        run(query, args.save, args.quiet)


def main():
    check_env()
    args = parse_args()

    if args.query:
        run(args.query, args.save, args.quiet)
    else:
        interactive(args)


if __name__ == "__main__":
    main()
