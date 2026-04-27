# 🔬 Agentic RAG — The Autonomous Researcher

An advanced **Agentic RAG (Retrieval-Augmented Generation)** system that goes beyond simple chatbots. It uses a **reasoning loop** to autonomously decide whether local knowledge is sufficient, fetch fresh web data when needed, and produce a comprehensive cited research report.

> **"Analyze the 2026 impact of carbon taxes on EV stocks"** → The agent searches your PDFs, realizes it needs fresher data, hits the web, synthesizes everything, checks for hallucinations, and delivers a cited report.

---

## 🏗️ Architecture

<img src="assets/architecture.png" width="800"/>

---

## ✨ Features

- **Agentic reasoning loop** — decides when local docs are enough vs. when to fetch web data
- **Local PDF ingestion** — drop any PDF into `sample_docs/` and it's searchable
- **ChromaDB vector store** — semantic search using free local embeddings
- **LLM relevance grading** — filters out irrelevant retrieved chunks
- **Hallucination checker** — verifies the report is grounded in sources
- **Dual web search** — Tavily (premium quality) with DuckDuckGo fallback (free)
- **Fully cited reports** — every claim linked to a source
- **100% Free** — Groq API + DuckDuckGo + local embeddings

---

## 🚀 Quick Start

### 1. Get a Free Groq API Key
Go to **[console.groq.com](https://console.groq.com/keys)** → Sign up → Create key

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ `sentence-transformers` downloads ~90MB model on first run (one-time only)

### 3. Add your documents
Drop `.pdf` or `.txt` files into `sample_docs/` — two example files are included.

### 4. Ingest documents
```bash
python ingest.py
```

### 5. Set API key & run

**PowerShell (Windows):**
```powershell
$env:GROQ_API_KEY="your_key_here"
python main.py
```

**CMD (Windows):**
```
set GROQ_API_KEY=your_key_here
python main.py
```

**Mac/Linux:**
```bash
export GROQ_API_KEY=your_key_here
python main.py
```

---

## 💻 Usage

```bash
# Interactive mode
python main.py

# Single query
python main.py "Analyze the 2026 impact of carbon taxes on EV stocks"

# Save report as markdown file
python main.py "Compare Tesla vs BYD investment thesis" --save

# Quiet mode (no step-by-step logs)
python main.py "What are the risks in lithium supply chains?" --quiet
```

---

## 📁 Project Structure

<img src="assets/project_structure.png" width="800"/>

---

## 🔧 Key Concepts Demonstrated

| Concept | Where |
|---|---|
| Agentic ReAct Loop | `graph.py` |
| LangGraph StateGraph | `graph.py` |
| RAG (Retrieval-Augmented Generation) | `agents/retriever.py` |
| Vector Embeddings (ChromaDB) | `agents/retriever.py` |
| LLM-as-Judge (grading) | `agents/grader.py` |
| Hallucination Detection | `agents/hallucination_checker.py` |
| Tool Use / Function Routing | `graph.py` conditional edges |
| Multi-source Synthesis | `agents/generator.py` |

---

## ⚙️ Optional: Better Web Search with Tavily

Tavily provides RAG-optimized web search (much better than DuckDuckGo for research). Free tier available:

1. Sign up at **[app.tavily.com](https://app.tavily.com)**
2. Get your free API key
3. Set it: `$env:TAVILY_API_KEY="tvly-..."`
4. Install: `pip install tavily-python`

The agent automatically uses Tavily when the key is set.

---

## 📦 Tech Stack

| Component | Tool | Cost |
|---|---|---|
| LLM | Groq (Llama 3.3 70B) | ✅ Free |
| Agentic Framework | LangGraph | ✅ Free |
| Vector DB | ChromaDB (local) | ✅ Free |
| Embeddings | sentence-transformers | ✅ Free |
| Web Search | DuckDuckGo / Tavily | ✅ Free tier |
| PDF Reading | pypdf | ✅ Free |

---

## 📄 License

MIT — free to use, extend, and build on.
