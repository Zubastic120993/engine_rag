import os
import gradio as gr
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# =====================================================
# 🚫 Disable Telemetry
# =====================================================
os.environ["LANGCHAIN_DISABLE_TELEMETRY"] = "true"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["POSTHOG_DISABLED"] = "true"
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

# =====================================================
# 🧠 Load Local Chroma RAG Store
# =====================================================
persist_dir = "engine_db"
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://127.0.0.1:11434"
)
db = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings
)

# =====================================================
# 🤖 Local LLM + Retrieval Chain
# =====================================================
llm = OllamaLLM(model="llama3", temperature=0)
retriever = db.as_retriever(search_kwargs={"k": 6}) 

# =====================================================
# 🧩 Custom Prompt Template
# =====================================================
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a professional MAN B&W ME-C Engine Assistant.\n\n"
        "If the user's question does not specify the exact engine model (e.g., S50ME-C, S60ME-C8, G70ME-C9), "
        "first ask which model they refer to before giving a detailed answer.\n\n"
        "Use ONLY the information from the provided context to answer.\n"
        "If exact values are not listed, summarize the relevant text and infer guidance values "
        "based on similar MAN B&W ME-C series engines.\n"
        "Never say 'I don't know' unless the context is completely unrelated.\n"
        "If the answer includes measurements, return them as clear bullet points with units.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    ),
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=False
)

# =====================================================
# 💬 Chat Function
# =====================================================
def chat_with_manuals(question, history):
    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"].strip()
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        return history, history
    except Exception as e:
        error_msg = f"⚠️ Error: {str(e)}"
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": error_msg})
        return history, history


# =====================================================
# 🧰 Build Gradio Interface
# =====================================================
with gr.Blocks(title="ME-C Engine Assistant") as demo:
    gr.Markdown(
        "## ⚓ ME-C Engine Smart Assistant\n"
        "Ask questions about your manuals and service letters — "
        "works fully offline using Ollama and your local RAG database."
    )

    chatbot = gr.Chatbot(height=400, type="messages")
    msg = gr.Textbox(placeholder="Ask about ME-C engine, service letters, etc...")
    clear = gr.Button("Clear Chat")

    msg.submit(chat_with_manuals, [msg, chatbot], [chatbot, chatbot]).then(
        lambda: "", None, msg
    )
    clear.click(lambda: [], None, chatbot, queue=False)

# =====================================================
# ▶️ Run App
# =====================================================
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)