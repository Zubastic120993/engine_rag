import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Ensure Ollama points to the correct daemon
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

# === 1️⃣ Find all PDFs in data/ folder ===
pdf_files = glob.glob("data/*.pdf")
if not pdf_files:
    raise FileNotFoundError("No PDF files found in data/ folder")

all_docs = []
for pdf_path in pdf_files:
    print(f"📄 Loading: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    all_docs.extend(loader.load())

# === 2️⃣ Split into smaller text chunks ===
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,          # safe size for embedding models
    chunk_overlap=120,
    separators=["\n\n", "\n", " ", ""],
)
chunks = splitter.split_documents(all_docs)

# Clean whitespace
for d in chunks:
    d.page_content = d.page_content.strip()

print(f"Total chunks: {len(chunks)}")

# === 3️⃣ Create embeddings & store in local Chroma DB ===
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://127.0.0.1:11434",
)

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="engine_db",
)
db.persist()

print(f"✅ {len(pdf_files)} PDF file(s) processed and stored in engine_db/")