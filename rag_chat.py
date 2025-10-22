import os
import urllib.parse
import gradio as gr
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

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
    model="mxbai-embed-large", base_url="http://127.0.0.1:11434"
)
db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# =====================================================
# 🧩 Memory + Conversational Chain
# =====================================================
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",  # ✅ prevents “multiple output key” error
)

llm = OllamaLLM(model="llama3", temperature=0)
retriever = db.as_retriever(search_kwargs={"k": 6})

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
)

# =====================================================
# 💬 Chat Function
# =====================================================
def chat_with_manuals(question, history):
    try:
        result = qa_chain.invoke({"question": question})
        answer = result["answer"].strip()

        # Build source markdown string safely
        sources_md = ""
        source_docs = result.get("source_documents", [])
        if source_docs:
            seen = set()
            for doc in source_docs:
                src = os.path.basename(doc.metadata.get("source", "Unknown.pdf"))
                page = doc.metadata.get("page", "?")
                key = (src, page)
                if key not in seen:
                    seen.add(key)
                    abs_path = os.path.abspath(os.path.join("data", src))
                    encoded_abs_path = urllib.parse.quote(abs_path)
                    file_uri = f"file://{encoded_abs_path}#page={page}"
                    sources_md += f"• [{src} — page {page}]({file_uri})\n"
            if sources_md:
                answer += "\n\n📎 **Sources:**\n" + sources_md

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        return history, history, sources_md, gr.update(value="")

    except Exception as e:
        error_msg = f"⚠️ Error: {str(e)}"
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": error_msg})
        return history, history, "", gr.update(value="")

# =====================================================
# 🔁 Reset Function — clears memory and chat
# =====================================================
def reset_chat():
    memory.clear()  # ✅ reset memory buffer
    return [], [], "", gr.update(value=""), gr.update(visible=False)

# =====================================================
# 🧰 Gradio Interface
# =====================================================
with gr.Blocks(title="ME-C Engine Assistant") as demo:
    gr.Markdown(
        "## ⚓ ME-C Engine Smart Assistant\nAsk questions about ME-C engine manuals and service letters — "
        "works fully offline using Ollama and your local RAG database."
    )

    chatbot = gr.Chatbot(height=400, type="messages")
    msg = gr.Textbox(placeholder="Ask about ME-C engine, service letters, etc...")
    clear = gr.Button("🧹 Clear Chat")
    sources_state = gr.State("")
    sources_box = gr.Markdown(visible=False)

    def show_sources(sources_md):
        """Show markdown string properly (not a list)."""
        if not sources_md:
            return gr.update(visible=False)
        md = f"### 📄 **Open Related Manuals in Preview:**\n{sources_md}"
        return gr.update(value=md, visible=True)

    msg.submit(
        chat_with_manuals,
        [msg, chatbot],
        [chatbot, chatbot, sources_state, msg],
    ).then(
        show_sources,
        [sources_state],
        [sources_box],
    )

    clear.click(
        reset_chat,
        None,
        [chatbot, sources_state, sources_box, msg],
        queue=False,
    )

# =====================================================
# ▶️ Run App
# =====================================================
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)