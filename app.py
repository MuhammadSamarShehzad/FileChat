import streamlit as st
from uuid import uuid4

from src.pipeline.core import (
	load_docs_from_pdf_bytes,
	build_graph_from_documents,
	ask_question,
)

st.set_page_config(page_title="FileChat", page_icon="📄", layout="centered")

# Session state
if "thread" not in st.session_state:
	st.session_state.thread = {"id": str(uuid4()), "messages": []}
if "graph" not in st.session_state:
	st.session_state.graph = None

# Upload
uploaded = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
if uploaded is not None:
	docs = load_docs_from_pdf_bytes(uploaded.name, uploaded.getvalue())
	st.session_state.graph = build_graph_from_documents(docs, k=4)

# Render history
for m in st.session_state.thread["messages"]:
	with st.chat_message("user" if m["role"] == "user" else "ai"):
		st.write(m["content"]) 

# Chat
prompt = st.chat_input("Ask a question")
if prompt:
	st.session_state.thread["messages"].append({"role": "user", "content": prompt})
	with st.chat_message("user"):
		st.write(prompt)

	if st.session_state.graph is None:
		answer = "Please upload a PDF first."
	else:
		answer = ask_question(st.session_state.graph, prompt, st.session_state.thread["id"])

	st.session_state.thread["messages"].append({"role": "assistant", "content": answer})
	with st.chat_message("ai"):
		st.write(answer)