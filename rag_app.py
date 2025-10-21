import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Ensure Ollama points to the correct daemon
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

# === 1) Find all PDFs in data/ folder ===
pdf_files = glob.glob("data/*.pdf")
if not pdf_files:
    raise FileNotFoundError("No PDF files found in data/ folder")

all_docs = []
for pdf_path in pdf_files:
    print(f"📄 Loading: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    all_docs.extend(loader.load())

# === 2) Split into small pieces (conservative to respect model context) ===
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,          # safer than 1000 for mxbai-embed-large
    chunk_overlap=120,
    separators=["\n\n", "\n", " ", ""],  # helps keep chunks tidy
)
chunks = splitter.split_documents(all_docs)

# Trim whitespace to avoid accidental oversized payloads
for d in chunks:
    d.page_content = d.page_content.strip()

print(f"Total chunks: {len(chunks)}")

# === 3) Create embeddings & store in local Chroma DB ===
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