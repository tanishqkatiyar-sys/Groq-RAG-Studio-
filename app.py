import streamlit as st

from main import (
    generate_response,
    PERSONALITIES,
    MODEL,
    API_KEY,
    load_embedding_model,
    load_vectorstore,
    load_llm,
    get_database_count,
    add_pdf_to_vectorstore,
    get_uploaded_documents,
    delete_document_from_vectorstore,
    clear_all_documents
)


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Groq RAG Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. MODERN GLASSMORPHIC OBSIDIAN THEME
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-canvas: #07090e;
    --bg-sidebar: #0c101a;
    --bg-surface: rgba(17, 24, 39, 0.72);
    --bg-surface-hover: rgba(30, 41, 59, 0.85);
    --bg-elevated: #161e31;
    --border-subtle: rgba(148, 163, 184, 0.14);
    --border-accent: rgba(99, 102, 241, 0.45);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-indigo: #6366f1;
    --accent-violet: #8b5cf6;
    --accent-cyan: #06b6d4;
    --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
    --success: #10b981;
    --warning: #f59e0b;
    --error: #ef4444;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 18px;
    --shadow-card: 0 8px 30px rgba(0, 0, 0, 0.35);
}

/* Global Typography & Canvas */
html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-canvas) !important;
    background-image: 
        radial-gradient(circle at 15% 10%, rgba(99, 102, 241, 0.08) 0%, transparent 35%),
        radial-gradient(circle at 85% 85%, rgba(6, 182, 212, 0.06) 0%, transparent 35%);
    color: var(--text-primary) !important;
    color-scheme: dark;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background: transparent !important;
}

/* Hide default footer & reduce top whitespace */
footer { visibility: hidden; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 5rem !important;
    max-width: 1100px;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"] > div {
    background-color: var(--bg-sidebar) !important;
}

/* Sidebar Tabs */
[data-testid="stSidebar"] [data-baseweb="tab-list"] {
    background-color: var(--bg-elevated);
    padding: 4px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-subtle);
    gap: 4px;
}

[data-testid="stSidebar"] [data-baseweb="tab"] {
    border-radius: 6px !important;
    padding: 6px 10px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    flex: 1;
    justify-content: center;
}

[data-testid="stSidebar"] [aria-selected="true"] {
    background: var(--accent-indigo) !important;
    color: #ffffff !important;
}

[data-testid="stSidebar"] [data-baseweb="tab-highlight"],
[data-testid="stSidebar"] [data-baseweb="tab-border"] {
    display: none !important;
}

/* Buttons */
.stButton button {
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border: 1px solid var(--border-subtle) !important;
}

.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
}

.stButton button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
}

.stButton button[kind="secondary"] {
    background-color: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
}

.stButton button[kind="secondary"]:hover {
    background-color: var(--bg-surface-hover) !important;
    border-color: var(--border-accent) !important;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background-color: var(--bg-elevated) !important;
    border: 1.5px dashed rgba(99, 102, 241, 0.4) !important;
    border-radius: var(--radius-md) !important;
    padding: 14px !important;
    transition: all 0.2s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-indigo) !important;
    background-color: rgba(99, 102, 241, 0.08) !important;
}

/* Top Hero Banner */
.hero-header {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    background: var(--bg-surface);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 18px 24px;
    margin-bottom: 20px;
    box-shadow: var(--shadow-card);
}

.hero-title {
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.hero-subtitle {
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin: 2px 0 0 0;
}

.status-pills {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-primary);
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: var(--success);
    box-shadow: 0 0 8px var(--success);
}

/* Welcome Showcase */
.welcome-box {
    background: var(--bg-surface);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 36px 32px;
    text-align: center;
    margin: 12px 0 24px 0;
    box-shadow: var(--shadow-card);
}

.welcome-box h2 {
    font-size: 1.75rem;
    font-weight: 700;
    margin-bottom: 10px;
    background: linear-gradient(to right, #ffffff, #a5b4fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.welcome-box p {
    color: var(--text-secondary);
    max-width: 580px;
    margin: 0 auto 8px auto;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    background: var(--bg-surface) !important;
    backdrop-filter: blur(10px);
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    padding: 18px 22px !important;
    margin-bottom: 14px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

[data-testid="stChatMessage"]:hover {
    border-color: var(--border-accent) !important;
}

/* Chat Input Bar */
[data-testid="stChatInput"] {
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45) !important;
}

[data-testid="stChatInput"] textarea {
    color: var(--text-primary) !important;
    font-size: 0.95rem !important;
}

/* Document Item Card in Sidebar */
.doc-item {
    display: flex;
    align-items: center;
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 8px 12px;
    font-size: 0.82rem;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Citation Source Card */
.source-card {
    background: var(--bg-elevated);
    border-left: 3px solid var(--accent-indigo);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 12px 14px;
    margin-bottom: 10px;
}

.source-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    font-weight: 600;
    color: #a5b4fc;
    margin-bottom: 6px;
}

.source-body {
    font-size: 0.84rem;
    color: var(--text-secondary);
    line-height: 1.55;
}

/* Selectbox & Expander Tweaks */
[data-baseweb="select"] > div,
[data-testid="stExpander"] {
    background-color: var(--bg-elevated) !important;
    border-color: var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
}

[data-testid="stMetric"] {
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    padding: 12px !important;
}

[data-testid="stMetricValue"] {
    color: #a5b4fc !important;
    font-size: 1.4rem !important;
}

/* Custom Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.25); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(148, 163, 184, 0.45); }
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. API KEY VALIDATION & RAG INITIALIZATION
# =====================================================

if not API_KEY:
    st.error("🔑 GROQ_API_KEY not found. Please configure your `.env` file.")
    st.stop()


@st.cache_resource
def initialize_rag():
    embedding_model = load_embedding_model()
    vectorstore = load_vectorstore(embedding_model)
    llm = load_llm()
    return vectorstore, llm


try:
    with st.spinner("⚡ Booting Groq RAG Engine..."):
        vectorstore, llm = initialize_rag()
        total_chunks = get_database_count(vectorstore)
        indexed_docs = get_uploaded_documents(vectorstore)
except Exception as e:
    st.error(f"RAG initialization failed: {e}")
    st.stop()


# =====================================================
# 4. SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "personality" not in st.session_state:
    st.session_state.personality = "Teacher"

if "upload_results" not in st.session_state:
    st.session_state.upload_results = []

if "processing_upload" not in st.session_state:
    st.session_state.processing_upload = False

if "queued_prompt" not in st.session_state:
    st.session_state.queued_prompt = None


# =====================================================
# 5. STARTER PROMPTS
# =====================================================

SUGGESTIONS = [
    ("📚 Core Concepts", "Explain the main concepts available in the documents."),
    ("📝 Executive Summary", "Summarize the most important information in the documents."),
    ("💡 Key Takeaways", "What are the top key insights from the uploaded documents?"),
    ("📋 Structured Outline", "Create a structured hierarchical outline of the document content."),
    ("❓ Topic Overview", "What are the primary topics and themes covered?"),
    ("🔍 Deep Dive", "Identify critical definitions, metrics, or facts in the documents."),
]


# =====================================================
# 6. SIDEBAR (ORGANIZED INTO TABS)
# =====================================================

with st.sidebar:
    st.markdown("""
    <div style="padding: 6px 0 12px 0;">
        <div style="display:flex; align-items:center; gap:10px;">
            <span style="font-size:1.6rem;">⚡</span>
            <div>
                <div style="font-weight:700; font-size:1.15rem; color:#f8fafc;">Groq RAG Studio</div>
                <div style="font-size:0.75rem; color:#64748b;">Ultra-Fast Document Intelligence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top Quick Actions
    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.rerun()
    with c2:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)

    # Organized Sidebar Tabs
    tab_docs, tab_params, tab_info = st.tabs(["📂 Knowledge", "⚙️ Tuning", "ℹ️ System"])

    # --- TAB 1: KNOWLEDGE BASE ---
    with tab_docs:
        st.caption("Upload PDF files to index into ChromaDB.")

        if st.session_state.processing_upload:
            st.info("🔄 Extracting & embedding PDFs...")

        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded_files:
            st.session_state.uploaded_files_buffer = uploaded_files

        if st.button(
            "⬆️ Index Selected PDFs",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.processing_upload or not st.session_state.get("uploaded_files_buffer")
        ):
            pending = st.session_state.get("uploaded_files_buffer", [])
            if pending:
                st.session_state.processing_upload = True
                st.session_state.upload_results = []
                st.session_state.pending_files = pending
                del st.session_state.uploaded_files_buffer
                st.rerun()

        # Processing loop
        if st.session_state.get("pending_files"):
            pending = st.session_state.pending_files
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []

            for idx, pdf_file in enumerate(pending):
                status_text.caption(f"Processing `{pdf_file.name}`...")
                try:
                    result = add_pdf_to_vectorstore(pdf_file, vectorstore)
                    results.append(result)
                except Exception as e:
                    results.append({
                        "status": "error",
                        "filename": pdf_file.name,
                        "error": f"{type(e).__name__}: {e}"
                    })
                progress_bar.progress((idx + 1) / len(pending))

            st.session_state.upload_results = results
            st.session_state.processing_upload = False
            del st.session_state.pending_files
            st.rerun()

        # Upload Feedback
        if st.session_state.upload_results:
            for res in st.session_state.upload_results:
                if res["status"] == "success":
                    st.success(f"**{res['filename']}** (+{res['chunks']} chunks)")
                elif res["status"] == "duplicate":
                    st.info(f"**{res['filename']}** already indexed.")
                else:
                    st.error(f"**{res['filename']}**: {res.get('error', 'Failed')}")

        st.divider()
        st.markdown(f"**Indexed Documents ({len(indexed_docs)})**")

        if indexed_docs:
            for filename in indexed_docs:
                col_name, col_del = st.columns([5, 1])
                with col_name:
                    st.markdown(f'<div class="doc-item" title="{filename}">📄 {filename}</div>', unsafe_allow_html=True)
                with col_del:
                    if st.button("✕", key=f"del_{filename}", help=f"Remove {filename}"):
                        delete_document_from_vectorstore(vectorstore, filename)
                        st.rerun()

            st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Wipe Knowledge Base", use_container_width=True):
                clear_all_documents(vectorstore)
                st.rerun()
        else:
            st.caption("No documents indexed yet.")

    # --- TAB 2: RETRIEVAL & PERSONA TUNING ---
    with tab_params:
        st.markdown("**🎭 Assistant Persona**")
        personality = st.selectbox(
            "Personality",
            options=list(PERSONALITIES.keys()),
            key="personality",
            label_visibility="collapsed"
        )
        st.caption(PERSONALITIES[personality])

        st.divider()
        st.markdown("**🔍 MMR Retrieval Parameters**")
        k = st.slider("Top-K Passages", 1, 10, 4, help="Final chunks passed to the LLM")
        fetch_k = st.slider("Candidate Pool (Fetch K)", 5, 30, 10, help="Initial chunks fetched before diversity filtering")
        lambda_mult = st.slider("Diversity Balance (λ)", 0.0, 1.0, 0.5, 0.1, help="0 = Max diversity, 1 = Max relevance")

        st.divider()
        st.markdown("**🧠 LLM Engine**")
        GROQ_MODELS = ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"]
        current_idx = GROQ_MODELS.index(MODEL) if MODEL in GROQ_MODELS else 0
        selected_model = st.selectbox("Model", options=GROQ_MODELS, index=current_idx, label_visibility="collapsed")
        if selected_model != MODEL:
            st.warning("Update `GROQ_MODEL` in `.env` and restart to switch models.")

    # --- TAB 3: SYSTEM METRICS ---
    with tab_info:
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Chunks", get_database_count(vectorstore))
        with m2:
            st.metric("PDFs", len(indexed_docs))

        st.markdown(f"""
        <div style="background: var(--bg-elevated); border: 1px solid var(--border-subtle); border-radius: 10px; padding: 14px; margin-top: 10px; font-size: 0.82rem; color: var(--text-secondary);">
            <div style="margin-bottom: 6px;"><span class="status-dot" style="display:inline-block; margin-right:6px;"></span><strong style="color:#f8fafc;">Vector Store:</strong> ChromaDB Online</div>
            <div style="margin-bottom: 6px;">⚡ <strong style="color:#f8fafc;">Active Model:</strong> <code>{MODEL}</code></div>
            <div>💬 <strong style="color:#f8fafc;">Session Turns:</strong> {len(st.session_state.messages)}</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.caption("Stack: Python • Streamlit • LangChain • ChromaDB • Groq LPU")


# =====================================================
# 7. MAIN WORKSPACE HEADER
# =====================================================

live_chunks = get_database_count(vectorstore)
live_docs = len(get_uploaded_documents(vectorstore))

st.markdown(f"""
<div class="hero-header">
    <div>
        <h1 class="hero-title">Groq RAG Assistant</h1>
        <p class="hero-subtitle">Context-grounded answers with page-level source citations</p>
    </div>
    <div class="status-pills">
        <span class="status-pill"><span class="status-dot"></span>{MODEL}</span>
        <span class="status-pill">🎭 {personality}</span>
        <span class="status-pill">📄 {live_docs} Docs · {live_chunks} Chunks</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 8. EMPTY STATE / SUGGESTION CARDS
# =====================================================

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-box">
        <h2>What would you like to discover today?</h2>
        <p>Upload your PDFs in the <strong>Knowledge</strong> tab on the left, or pick a starter prompt below to query your indexed documents.</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    for idx, (label, query_text) in enumerate(SUGGESTIONS):
        with cols[idx % 3]:
            if st.button(label, key=f"sug_{idx}", use_container_width=True):
                st.session_state.queued_prompt = query_text
                st.rerun()


# =====================================================
# 9. CHAT HISTORY RENDERING
# =====================================================

for message in st.session_state.messages:
    is_user = message["role"] == "user"
    with st.chat_message(message["role"], avatar="🧑‍💻" if is_user else "⚡"):
        st.markdown(message["content"])

        if not is_user and message.get("sources"):
            with st.expander(f"📑 View {len(message['sources'])} Cited Passages"):
                for i, src in enumerate(message["sources"], 1):
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">
                            <span>#{i} · {src['source']}</span>
                            <span>Page {src['page']}</span>
                        </div>
                        <div class="source-body">{src['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)


# =====================================================
# 10. CHAT INPUT & RAG PIPELINE EXECUTION
# =====================================================

user_input = st.chat_input("Ask anything about your uploaded PDFs...")
prompt = st.session_state.queued_prompt or user_input

if st.session_state.queued_prompt:
    st.session_state.queued_prompt = None

if prompt:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="⚡"):
        try:
            with st.status("Retrieving context & synthesizing...", expanded=True) as status:
                st.write("🔍 Searching vector embeddings via MMR...")
                result = generate_response(
                    query=prompt,
                    personality=personality,
                    vectorstore=vectorstore,
                    llm=llm,
                    k=k,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult
                )
                st.write(f"📄 Matched {result['retrieved_count']} relevant chunks")
                status.update(label="Response ready", state="complete", expanded=False)

            answer = result.get("answer", "").strip()
            if not answer:
                st.warning("⚠️ Model returned an empty response. Try rephrasing or checking API status.")
                st.session_state.messages.pop()
            else:
                st.markdown(answer)

                if result.get("sources"):
                    with st.expander(f"📑 View {result['retrieved_count']} Cited Passages"):
                        for i, src in enumerate(result["sources"], 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <div class="source-header">
                                    <span>#{i} · {src['source']}</span>
                                    <span>Page {src['page']}</span>
                                </div>
                                <div class="source-body">{src['content']}</div>
                            </div>
                            """, unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": result["sources"]
                })

        except Exception as e:
            st.session_state.messages.pop()
            st.error(f"❌ Generation error: {e}")