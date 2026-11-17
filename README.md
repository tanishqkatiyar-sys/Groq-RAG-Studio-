# Groq RAG Studio ⚡

> **Ultra-fast document intelligence powered by Groq LPU, ChromaDB, and LangChain.**  
> Upload PDFs, ask questions, get cited answers in milliseconds.

---

## 🎯 Overview

**Groq RAG Studio** is a production-ready Retrieval-Augmented Generation (RAG) application built with Streamlit. It transforms your PDF documents into a searchable knowledge base and answers questions with **page-level source citations** — all running on Groq's lightning-fast LPU inference.

| Feature | Implementation |
|---------|----------------|
| **LLM Inference** | Groq (Llama 3.1, Mixtral, Gemma 2) |
| **Vector Store** | ChromaDB (persistent, local) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local, CPU) |
| **PDF Extraction** | PyMuPDF (fitz) — 50–100× faster than PyPDF |
| **Chunking** | RecursiveCharacterTextSplitter (1000/200) |
| **Retrieval** | MMR (Maximal Marginal Relevance) with diversity control |
| **UI** | Streamlit with glassmorphic Obsidian theme |

---

## ✨ Key Features

| Capability | Description |
|------------|-------------|
| 📄 **Multi-PDF Knowledge Base** | Drag & drop multiple PDFs, auto-deduplicated by content hash |
| ⚡ **Sub-second Retrieval** | Groq LPU + ChromaDB = answers in ~200ms |
| 📑 **Cited Answers** | Every response shows exact source passages with page numbers |
| 🎭 **6 Personalities** | Teacher, Developer, Mentor, Researcher, Interviewer, General Assistant |
| 🔍 **MMR Retrieval Tuning** | Adjust `k`, `fetch_k`, and diversity (`λ`) in the sidebar |
| 🗑️ **Full Document Management** | Delete individual PDFs or wipe the entire knowledge base |
| 💬 **Session Persistence** | Chat history survives reruns; export-ready |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Groq API Key](https://console.groq.com/keys) (free tier available)

### Installation

```bash
# 1. Clone & enter
git clone <your-repo-url>
cd Groq-RAG-Studio

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Run the App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
Groq-RAG-Studio/
├── app.py                 # Streamlit UI (theme, chat, sidebar, upload)
├── main.py                # Core RAG logic (loaders, retrieval, generation)
├── create_database.py     # (Optional) Bulk PDF ingestion script
├── document_loaders/      # Custom loader utilities (if any)
├── vector_store/          # Vector store helpers (if any)
├── chroma_db/             # Persistent ChromaDB (auto-created, gitignored)
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project metadata
├── .env.example           # Environment template
├── .gitignore
└── README.md
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# Required
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx

# Optional (defaults shown)
GROQ_MODEL=openai/gpt-oss-120b          # Model: llama-3.1-8b-instant, mixtral-8x7b-32768, gemma2-9b-it
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Retrieval Parameters (Sidebar → Tuning)

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| **Top-K** | 1–10 | 4 | Final chunks sent to LLM |
| **Fetch K** | 5–30 | 10 | Initial candidate pool before MMR |
| **Diversity (λ)** | 0.0–1.0 | 0.5 | 0 = max diversity, 1 = max relevance |

---

## 🎨 Personalities

| Personality | Best For |
|-------------|----------|
| **Teacher** | Step-by-step explanations, learning new topics |
| **Developer** | Code snippets, debugging, best practices |
| **Mentor** | Career guidance, structured advice |
| **Researcher** | Analytical breakdowns, fact vs. uncertainty |
| **Interviewer** | Interview-style Q&A, technical screening prep |
| **General Assistant** | Friendly, balanced, everyday help |

---

## 📸 Screenshots

### Knowledge
![Knowledge](screenshots\Screenshot 2026-11-17 131135.png)

### Tuning
![Tuning](screenshots\Screenshot 2026-11-17 131210.png)

### System
![System](screenshots\Screenshot 2026-11-17 131244.png)

## 🛠️ Development

### Run Tests (if added)
```bash
pytest -v
```

### Format Code
```bash
black app.py main.py
isort app.py main.py
```

### Type Check
```bash
mypy app.py main.py
```

---

## 📦 Dependencies

**Core** (`requirements.txt`):
```
streamlit>=1.40.0
groq>=0.11.0
python-dotenv>=1.0.0
langchain>=0.3.0
langchain-community>=0.3.0
langchain-core>=0.3.0
langchain-text-splitters>=0.3.0
langchain-huggingface>=0.1.0
langchain-groq>=0.2.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
pymupdf>=1.24.0
pypdf>=4.0.0
```

---

## 🔒 Privacy & Security

- **100% Local Embeddings** — Documents never leave your machine for embedding
- **Groq API Only for Generation** — Only query + retrieved chunks sent to Groq
- **No Telemetry** — Zero tracking, analytics, or external calls beyond Groq
- **Content-Hash Deduplication** — Identical PDFs skipped automatically

---

## 🗺️ Roadmap

- [ ] Hybrid search (BM25 + vector)
- [ ] Multi-collection support (projects/workspaces)
- [ ] Streaming token-by-token responses
- [ ] PDF OCR fallback for scanned documents
- [ ] Export chat as Markdown/PDF
- [ ] Dockerfile + docker-compose
- [ ] Unit & integration tests

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-thing`)
3. Commit your changes (`git commit -m 'Add amazing thing'`)
4. Push to the branch (`git push origin feature/amazing-thing`)
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Groq** — For the world's fastest LLM inference
- **LangChain** — For the RAG orchestration framework
- **ChromaDB** — For the lightweight vector database
- **Streamlit** — For the delightful Python-to-UI framework
- **PyMuPDF** — For blazing-fast PDF extraction

---

<div align="center">

**Built with ❤️ for developers who value speed and citations.**

[⭐ Star this repo] • [🐛 Report Bug] • [💡 Request Feature]

</div>