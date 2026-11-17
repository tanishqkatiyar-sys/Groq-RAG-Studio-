# Groq RAG Studio ⚡

<p align="center">
  <strong>Ultra-fast document intelligence powered by Groq LPU, ChromaDB, and LangChain.</strong>
  <br>
  Upload PDFs, ask questions, and get cited answers in milliseconds.
  <br><br>
  <a href="https://2pakglvn7v3qjr4zl8jfh4.streamlit.app/">
    🚀 <strong>Live Demo</strong>
  </a>
</p>

---

## 🎯 Overview

**Groq RAG Studio** is a production-ready Retrieval-Augmented Generation (RAG) application built with Streamlit. It transforms your PDF documents into a searchable knowledge base and answers questions with **page-level source citations** — all running on Groq's fast LPU inference.

| Feature | Implementation |
|---------|----------------|
| **LLM Inference** | Groq (Llama 3.1, Mixtral, Gemma 2) |
| **Vector Store** | ChromaDB (persistent, local) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local, CPU) |
| **PDF Extraction** | PyMuPDF (fitz) |
| **Chunking** | RecursiveCharacterTextSplitter (1000/200) |
| **Retrieval** | MMR (Maximal Marginal Relevance) with diversity control |
| **UI** | Streamlit with glassmorphic Obsidian theme |

---

## ✨ Key Features

| Capability | Description |
|------------|-------------|
| 📄 **Multi-PDF Knowledge Base** | Upload multiple PDFs, automatically deduplicated by content hash |
| ⚡ **Fast Retrieval** | Groq LPU + ChromaDB for fast document question answering |
| 📑 **Cited Answers** | Responses include source passages and page numbers |
| 🎭 **6 Personalities** | Teacher, Developer, Mentor, Researcher, Interviewer, General Assistant |
| 🔍 **MMR Retrieval Tuning** | Adjust `k`, `fetch_k`, and diversity (`λ`) in the sidebar |
| 🗑️ **Document Management** | Delete individual PDFs or clear the knowledge base |
| 💬 **Session Persistence** | Chat history survives Streamlit reruns |

---

## 🚀 Live Demo

Try the deployed application:

**[🌐 Open Groq RAG Studio](https://2pakglvn7v3qjr4zl8jfh4.streamlit.app/)**

---

## 📸 Screenshots

### Knowledge Base
![Knowledge](screenshots/Screenshot%202026-11-17%20131135.png)

### Retrieval Tuning
![Tuning](screenshots/Screenshot%202026-11-17%20131210.png)

### System Information
![System](screenshots/Screenshot%202026-11-17%20131244.png)

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLM Provider:** Groq
- **LLM Framework:** LangChain
- **Vector Database:** ChromaDB
- **Embedding Model:** Sentence Transformers
- **PDF Processing:** PyMuPDF
- **Language:** Python

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Groq API Key](https://console.groq.com/keys) (free tier available)

### Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>

# 2. Enter the project directory
cd Groq-RAG-Studio

# 3. Create a virtual environment
python -m venv venv

# 4. Activate the environment

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Configure environment variables
cp .env.example .env
```

Add your Groq API key to the `.env` file:

```env
GROQ_API_KEY=your_actual_groq_api_key
```

### Run the App

```bash
streamlit run app.py
```

Open your browser:

```text
http://localhost:8501
```

---

## 📁 Project Structure

```text
Groq-RAG-Studio/
│
├── app.py                  # Streamlit UI
├── main.py                 # Core RAG logic
├── create_database.py      # Optional PDF ingestion script
│
├── screenshots/            # Application screenshots
│   ├── Screenshot 2026-11-17 131135.png
│   ├── Screenshot 2026-11-17 131210.png
│   └── Screenshot 2026-11-17 131244.png
│
├── document_loaders/       # Custom document loaders
├── vector_store/           # Vector store utilities
├── chroma_db/              # Persistent ChromaDB (auto-created)
│
├── requirements.txt        # Python dependencies
├── pyproject.toml           # Project metadata
├── .env.example             # Environment template
├── .gitignore
└── README.md
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Required
GROQ_API_KEY=your_groq_api_key

# Optional
GROQ_MODEL=openai/gpt-oss-120b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Streamlit Cloud Secrets

For deployment on Streamlit Community Cloud:

1. Open your app dashboard.
2. Navigate to **Settings → Secrets**.
3. Add:

```toml
GROQ_API_KEY = "your_actual_groq_api_key"
```

4. Save and wait for the changes to propagate.

**Security:** Never upload your actual `.env` file or API key to GitHub.

### Retrieval Parameters (Sidebar → Tuning)

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| **Top-K** | 1–10 | 4 | Final chunks sent to the LLM |
| **Fetch K** | 5–30 | 10 | Initial candidate pool before MMR |
| **Diversity (λ)** | 0.0–1.0 | 0.5 | 0 = maximum diversity, 1 = maximum relevance |

---

## 🎨 AI Personalities

| Personality | Best For |
|-------------|----------|
| **Teacher** | Step-by-step explanations and learning new topics |
| **Developer** | Code snippets, debugging, and best practices |
| **Mentor** | Career guidance and structured advice |
| **Researcher** | Analytical breakdowns and fact-based answers |
| **Interviewer** | Interview-style questions and technical screening |
| **General Assistant** | Friendly and balanced everyday assistance |

---

## 🔄 How It Works

1. **Upload PDF:** Upload one or multiple PDF documents.
2. **Text Extraction:** PyMuPDF extracts text from the uploaded documents.
3. **Text Chunking:** LangChain splits extracted text into manageable chunks.
4. **Embedding Generation:** Sentence Transformers converts chunks into vector embeddings.
5. **Vector Storage:** ChromaDB stores the embeddings for similarity search.
6. **Question Retrieval:** MMR retrieves relevant document chunks.
7. **LLM Generation:** Groq generates an answer using the retrieved context.
8. **Source Citations:** The application displays relevant source passages and page numbers.

---

## 📦 Dependencies

Core dependencies from `requirements.txt`:

```text
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

- **Local Embeddings:** Document embeddings are generated locally.
- **Groq API for Generation:** Queries and retrieved context are sent to Groq for answer generation.
- **No Application Telemetry:** No custom tracking or analytics are included.
- **Content-Hash Deduplication:** Duplicate PDFs are skipped automatically.

---

## 🗺️ Roadmap

- [ ] Hybrid search (BM25 + vector search)
- [ ] Multi-collection support (projects/workspaces)
- [ ] Streaming token-by-token responses
- [ ] PDF OCR fallback for scanned documents
- [ ] Export chat as Markdown/PDF
- [ ] Dockerfile + Docker Compose
- [ ] Unit and integration tests

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a feature branch:

   ```bash
   git checkout -b feature/amazing-thing
   ```

3. Commit your changes:

   ```bash
   git commit -m "Add amazing feature"
   ```

4. Push to your branch:

   ```bash
   git push origin feature/amazing-thing
   ```

5. Open a Pull Request.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **[Groq](https://groq.com/)** — Fast LLM inference.
- **[LangChain](https://www.langchain.com/)** — RAG orchestration framework.
- **[ChromaDB](https://www.trychroma.com/)** — Vector database.
- **[Streamlit](https://streamlit.io/)** — Python application framework.
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** — PDF text extraction.

---

<p align="center">
  <strong>Built with ❤️ for developers who value speed and citations.</strong>
  <br><br>
  🚀 <a href="https://2pakglvn7v3qjr4zl8jfh4.streamlit.app/">Live Demo</a>
</p>