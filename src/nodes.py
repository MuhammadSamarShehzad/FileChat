from .state import QAState
from .LLM import llm
import json
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# ---- Nodes ----
def gen_queries(state: QAState):
    prompt = f"""
You are a query rewriter for document retrieval.

Document Start:
{state['doc_text']}

User Query: {state['query']}

Task: Generate 2–5 optimized rephrasings of the user query.
.
Guidelines:
- Infer document type (resume, research paper, notes, generic).
- Keep each query ≤12 words.
- Include lexical/structural variations (symbols, abbreviations, formats).
- For email include variants with "@", ".com", "email address", "contact".
- For social profiles include domain forms (e.g., "github.com", "linkedin.com/in").
- Output JSON only, in the exact format below.

Example Output:
{{"queries": ["query1", "query2", "query3"]}}
"""
    raw = llm.invoke(prompt).content.strip()
    try:
        qs = json.loads(raw).get("queries", [])
    except Exception:
        qs = [q.strip("-• ") for q in raw.splitlines() if q.strip()]
    state["queries"] = qs[:5]
    return {"queries":qs}




def gen_bm25_terms(state: QAState):
    prompt = f"""
You are a keyword extractor for BM25 search.

User Query: {state['query']}

Task: Produce 3–9 minimal keywords/tokens to maximize lexical recall.

Guidelines:
- Lowercase all.
- Include field labels & symbols where relevant (email, "@", phone, linkedin, github, website).
- Include domain forms when relevant (github.com, linkedin.com/in, gmail.com).
- Output JSON only, exactly:

{{"terms": ["term1", "term2", "term3"]}}
"""
    raw = llm.invoke(prompt).content.strip()
    try:
        terms = json.loads(raw).get("terms", [])
    except Exception:
        terms = [t.strip("-• ") for t in raw.splitlines() if t.strip()]
    # normalize tokens
    terms = [t.lower().strip() for t in terms if t]
    # dedupe
    seen, out = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    state["bm25_terms"] = out[:9]
    return state



def retrieve(state: QAState, vs: FAISS, bm25: BM25Retriever, raw_docs):
    # Dense search (FAISS)
    dense_hits = []
    for q in [state["query"]] + state.get("queries", []):
        dense_hits.extend(vs.similarity_search(q, k=3))

    # BM25 search
    bm25_hits = bm25.invoke(state["query"])

    # Merge results (unique text only)
    seen, merged = set(), []
    for d in dense_hits + bm25_hits:
        text = d.page_content.strip()
        if text and text not in seen:
            seen.add(text)
            merged.append(text)

    state["retrieved"] = merged[:15]
    return state




# def gen_answer(state: QAState):
#     context = "\n\n".join(state["retrieved"]) if state.get("retrieved") else ""
#     prompt = f"""
# You are given a user question and related context retrieved from documents.

# Question: {state['query']}
# Context:
# {context}

# Task: Provide a direct, accurate, and concise answer based only on the context.
# If not in context, reply with "Not found in document".
# Answer:
# """
#     state["answer"] = llm.invoke(prompt).content.strip()
#     return state


def gen_answer(state: QAState):
    context = "\n\n".join(state["retrieved"]) if state.get("retrieved") else ""
    prompt = f"""
You are given a user question and related context retrieved from documents.

Question: {state['query']}
Context:
{context}

Task: Provide a direct, accurate, and concise answer based only on the context.
If not in context, reply with "Not found in document".
Answer:
"""
    # Create a proper message object instead of just a string
    from langchain_core.messages import AIMessage
    ai_message = AIMessage(content=llm.invoke(prompt).content.strip())
    state["answer"] = [ai_message]  # Wrap in list as expected by add_messages
    return state