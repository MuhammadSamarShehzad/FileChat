import streamlit as st
from uuid import uuid4
import hashlib
import logging

from src.pipeline.core import (
    load_docs_from_pdf_bytes,
    build_graph_from_documents,
    ask_question,
)
from src.db.database import db
from src.utils.logger import setup_logging

# Initialize logging
setup_logging(log_file="app.log")
logger = logging.getLogger(__name__)
logger.info("FileChat application started")

st.set_page_config(page_title="FileChat", page_icon="📄", layout="wide")


def load_pdf_data_and_graph(pdf_id: int):
    """Load PDF chunks and rebuild graph from stored data."""
    pdf_chunks = db.get_pdf_chunks(pdf_id)
    if pdf_chunks:
        from langchain_core.documents import Document
        docs = [Document(page_content=chunk) for chunk in pdf_chunks]
        return build_graph_from_documents(docs, k=4)
    return None


# Initialize session state
if "thread" not in st.session_state:
    # Always start with a fresh chat on app restart
    st.session_state.thread = {"id": str(uuid4()), "messages": []}
    st.session_state.pdf_id = None
    st.session_state.graph = None
    st.session_state.thread_saved = False  # Track if thread has been saved to DB

    # Clean up any empty threads from previous sessions
    db.cleanup_empty_threads()

# Sidebar for chat threads
with st.sidebar:
    st.title("📚 Chat Threads")

    # New Chat button
    if st.button("🆕 New Chat", use_container_width=True):
        st.session_state.thread = {"id": str(uuid4()), "messages": []}
        st.session_state.graph = None
        st.session_state.pdf_id = None
        st.session_state.thread_saved = False  # Reset saved flag for new thread
        # Don't save empty thread to database

    st.divider()

    # Load existing threads
    threads = db.get_all_threads()

    if threads:
        for thread in threads:
            button_key = f"thread_{thread['id']}"
            pdf_name = thread['pdf_name'][:20] + "..." if len(thread['pdf_name']) > 20 else thread['pdf_name']
            message_count = thread['message_count']

            if st.button(
                f"💬 {pdf_name}\n📝 {message_count} messages",
                key=button_key,
                use_container_width=True,
                help=f"Created: {thread['created_at']}"
            ):
                # Load this thread
                st.session_state.thread = {"id": thread['id'], "messages": []}
                st.session_state.pdf_id = db.get_thread_pdf_id(thread['id'])

                # Load messages and rebuild graph
                existing_messages = db.get_chat_history(thread['id'])
                st.session_state.thread["messages"] = existing_messages

                if st.session_state.pdf_id:
                    st.session_state.graph = load_pdf_data_and_graph(st.session_state.pdf_id)
    else:
        st.info("No chat threads yet. Start a new chat!")

# Main content area
st.title("📄 FileChat")

# Show current thread info
if st.session_state.pdf_id:
    pdf_info = db.get_pdf_chunks(st.session_state.pdf_id)
    if pdf_info:
        st.info(f"📎 Current PDF: {pdf_info[0][:50]}..." if len(pdf_info[0]) > 50 else f"📎 Current PDF: {pdf_info[0]}")

# Upload (only if no PDF is loaded)
if st.session_state.pdf_id is None:
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
    if uploaded is not None:
        # Generate content hash for PDF
        content_hash = hashlib.md5(uploaded.getvalue()).hexdigest()

        # Load and process documents
        docs = load_docs_from_pdf_bytes(uploaded.name, uploaded.getvalue())
        st.session_state.graph = build_graph_from_documents(docs, k=4)

        # Store PDF in database
        chunks = [doc.page_content for doc in docs]
        pdf_id = db.store_pdf(uploaded.name, content_hash, chunks)
        st.session_state.pdf_id = pdf_id

        # Update chat thread with PDF reference
        db.create_chat_thread(st.session_state.thread["id"], pdf_id)
        st.session_state.thread_saved = True  # Mark thread as saved since it now has PDF content

# Render chat history
chat_container = st.container()
with chat_container:
    for m in st.session_state.thread["messages"]:
        with st.chat_message("user" if m["role"] == "user" else "ai"):
            st.write(m["content"])

# Chat input
prompt = st.chat_input("Ask a question")
if prompt:
    # Save thread to database only when it has content (first message)
    if not st.session_state.thread_saved:
        db.create_chat_thread(st.session_state.thread["id"])
        st.session_state.thread_saved = True

    # Add user message to database
    db.add_message(st.session_state.thread["id"], "user", prompt)
    st.session_state.thread["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.graph is None:
        answer = "Please upload a PDF first."
    else:
        answer = ask_question(st.session_state.graph, prompt, st.session_state.thread["id"])

    # Add AI response to database
    db.add_message(st.session_state.thread["id"], "assistant", answer)
    st.session_state.thread["messages"].append({"role": "assistant", "content": answer})

    with st.chat_message("ai"):
        st.write(answer)
