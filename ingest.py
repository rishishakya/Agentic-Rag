"""
Ingest documents into ChromaDB vector store.

Usage:
    python ingest.py                        # ingest from sample_docs/
    python ingest.py --folder my_pdfs/     # custom folder
    python ingest.py --reset               # clear DB and re-ingest
"""

import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument("--folder", default="./sample_docs", help="Folder with PDFs/TXTs")
    parser.add_argument("--reset",  action="store_true",     help="Clear DB before ingesting")
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("⚠️  GROQ_API_KEY not set (needed for grading). Set it before running main.py")

    if args.reset:
        import shutil
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
            print("🗑️  Cleared existing ChromaDB")

    from agents.retriever import ingest_documents

    print(f"📂 Ingesting documents from: {args.folder}")
    count = ingest_documents(args.folder)

    if count == 0:
        print(f"\n⚠️  No new documents ingested.")
        print(f"   Make sure {args.folder}/ contains .txt or .pdf files.")
    else:
        print(f"\n✅ Ingested {count} new chunks into ChromaDB")
        print("   Ready to run: python main.py")


if __name__ == "__main__":
    main()
