import streamlit as st
import _osx_support
from src.graph import build_graph
from src.store import build_store

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    save_path = os.path.join("data", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    vs, bm25, raw_docs, intro = build_store(save_path)
    graph, checkpointer = build_graph(vs, bm25, raw_docs)
    st.session_state["graph"] = graph
    st.session_state["intro"] = intro

# Chat
if not st.session_state["message_history"]:
    st.info("NO messages found")
else:
    for msg in st.session_state["message_history"]:
        with st.chat_message(msg["role"]):
            st.text(msg["content"])

prompt = st.chat_input("Ask something about the PDF")
if prompt and "graph" in st.session_state:
    st.session_state["message_history"].append({"role": "human", "content": prompt})
    with st.chat_message("human"):
        st.write(prompt)

    config = {"configurable": {"thread_id": "1"}}
    result = st.session_state["graph"].invoke({"doc_text": st.session_state["intro"], "query": prompt}, config=config)

    # Extract the actual answer content from the message
    if result.get("answer") and len(result["answer"]) > 0:
        answer_content = result["answer"][0].content
    else:
        answer_content = "No answer generated"

    st.session_state["message_history"].append({"role": "ai", "content": answer_content})
    with st.chat_message("ai"):
        st.write(answer_content)
