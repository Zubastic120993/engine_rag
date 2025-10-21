# ==============================================================
# 🚫 Disable all telemetry BEFORE imports
# ==============================================================
import os
os.environ["LANGCHAIN_DISABLE_TELEMETRY"] = "true"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["POSTHOG_DISABLED"] = "true"
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

# ==============================================================
# 🧠 Imports
# ==============================================================
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# ==============================================================
# ⚙️ Load existing Chroma database
# ==============================================================
persist_directory = "engine_db"

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://127.0.0.1:11434",
)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
)

# ==============================================================
# 💬 Local LLM (deterministic)
# ==============================================================
llm = OllamaLLM(
    model="llama3",
    temperature=0,  # ensures same result every run
)

# ==============================================================
# 🔍 Ask Question Function
# ==============================================================
def ask_question(query: str):
    # Compatible retriever (no fetch_k)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    results = db.similarity_search(query, k=3)  # direct deterministic call

    # Merge retrieved chunks
    context = "\n\n".join([doc.page_content.strip() for doc in results])
    prompt = (
        f"Answer the following question based only on the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    response = llm.invoke(prompt)

    print("\n🧠 QUESTION:", query)
    print("💡 ANSWER:\n", response)


# ==============================================================
# ▶️ Example
# ==============================================================
if __name__ == "__main__":
    query = "What is the recommended exhaust valve clearance for ME-C engine?"
    ask_question(query)