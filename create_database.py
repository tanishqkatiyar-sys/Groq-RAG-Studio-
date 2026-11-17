
import os
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

PDF_PATH = BASE_DIR / "document_loaders" / "FULLBOOKPHYSICS.pdf"

CHROMA_PATH = str(BASE_DIR / "chroma_db")

COLLECTION_NAME = "pdf_rag"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# =====================================================
# 2. VALIDATE PDF
# =====================================================

if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

with open(PDF_PATH, "rb") as file:
    pdf_bytes = file.read()

if not pdf_bytes.startswith(b"%PDF-"):
    raise ValueError("Invalid PDF file.")

file_hash = hashlib.sha256(pdf_bytes).hexdigest()

print("PDF found:", PDF_PATH.name)
print("Document hash:", file_hash)


# =====================================================
# 3. LOAD PDF
# =====================================================

print("\nLoading PDF...")

loader = PyPDFLoader(str(PDF_PATH))

docs = loader.load()

if not docs:
    raise ValueError("No readable pages found.")

print("Total PDF pages:", len(docs))


# =====================================================
# 4. ADD DOCUMENT METADATA
# =====================================================

for doc in docs:

    page_number = doc.metadata.get("page", 0)

    doc.metadata = {
        "source": PDF_PATH.name,
        "filename": PDF_PATH.name,
        "page": int(page_number) + 1,
        "document_id": file_hash
    }


# =====================================================
# 5. SPLIT DOCUMENT INTO CHUNKS
# =====================================================

print("\nSplitting PDF into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks = splitter.split_documents(docs)

chunks = [
    chunk for chunk in chunks
    if chunk.page_content.strip()
]

if not chunks:
    raise ValueError("No extractable text found.")

print("Total chunks:", len(chunks))


# =====================================================
# 6. LOAD EMBEDDING MODEL
# =====================================================

print("\nLoading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Embedding model loaded successfully.")


# =====================================================
# 7. CREATE / LOAD CHROMA DATABASE
# =====================================================

print("\nConnecting to ChromaDB...")

Path(CHROMA_PATH).mkdir(
    parents=True,
    exist_ok=True
)

vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME
)


# =====================================================
# 8. DUPLICATE DETECTION
# =====================================================

existing = vectorstore._collection.get(
    where={"document_id": file_hash},
    include=["metadatas"]
)

if existing.get("ids"):

    print("\nPDF already exists in ChromaDB.")
    print("Skipping duplicate insertion.")

else:

    # Deterministic IDs for every chunk
    ids = [
        f"{file_hash}_{index}"
        for index in range(len(chunks))
    ]

    print("\nGenerating embeddings and storing chunks...")

    batch_size = 500

    for start in range(0, len(chunks), batch_size):

        end = start + batch_size

        vectorstore.add_documents(
            documents=chunks[start:end],
            ids=ids[start:end]
        )

        print(
            f"Stored {min(end, len(chunks))}/{len(chunks)} chunks"
        )


# =====================================================
# 9. VERIFY DATABASE
# =====================================================

total_chunks = vectorstore._collection.count()

print("\n====================================")
print("DATABASE CREATION VERIFICATION")
print("====================================")

print("Collection:", COLLECTION_NAME)
print("PDF:", PDF_PATH.name)
print("PDF pages:", len(docs))
print("PDF chunks:", len(chunks))
print("Total database chunks:", total_chunks)

stored = vectorstore._collection.get(
    where={"document_id": file_hash},
    include=["metadatas"]
)

print("Stored chunks for this PDF:", len(stored["ids"]))

print("\nDatabase creation completed successfully!")
