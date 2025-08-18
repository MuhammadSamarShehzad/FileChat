from .state import QAState
from .LLM import llm
import json
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# ---- Nodes ----
def gen_queries(state: QAState):
	prompt = """
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
	{"queries": ["query1", "query2", "query3"]}
	"""
	raw = llm.invoke(prompt).content.strip()
	try:
		qs = json.loads(raw).get("queries", [])
	except Exception:
		qs = [q.strip("-• ") for q in raw.splitlines() if q.strip()]
	state["queries"] = qs[:5]
	return {"queries":qs}


def gen_bm25_terms(state: QAState):
	prompt = """
	You are a keyword extractor for BM25 search.

	User Query: {state['query']}

	Task: Produce 3–9 minimal keywords/tokens to maximize lexical recall.

	Guidelines:
	- Lowercase all.
	- Include field labels & symbols where relevant (email, "@", phone, linkedin, github, website).
	- Include domain forms when relevant (github.com, linkedin.com/in, gmail.com).
	- Output JSON only, exactly:

	{"terms": ["term1", "term2", "term3"]}
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
	return {"bm25_terms": out[:9]}


def retrieve(state: QAState, vs: FAISS, bm25: BM25Retriever, raw_docs):
	# Build expanded queries: original + rewrites + keyword terms
	queries = [state["query"]] + state.get("queries", [])
	if state.get("bm25_terms"):
		queries.append(" ".join(state["bm25_terms"]))

	# Dense search (FAISS)
	dense_hits = []
	for q in queries:
		dense_hits.extend(vs.similarity_search(q, k=3))

	# BM25 search (each query variant)
	bm25_hits = []
	for q in queries:
		bm25_hits.extend(bm25.invoke(q))

	# Merge results (unique text only)
	seen, merged = set(), []
	for d in dense_hits + bm25_hits:
		text = d.page_content.strip()
		if text and text not in seen:
			seen.add(text)
			merged.append(text)

	state["retrieved"] = merged[:15]
	return {"retrieved": merged[:15]}


def gen_answer(state: QAState):
	context = "\n\n".join(state["retrieved"]) if state.get("retrieved") else ""
	# If no context, return deterministic fallback without calling the LLM
	if not context:
		from langchain_core.messages import AIMessage
		return {"answer": [AIMessage(content="Not found in document")]}

	prompt_system = (
		"You answer strictly and only from the provided context. "
		"If the answer is not present in the context, reply exactly: Not found in document."
	)
	human_payload = f"Question: {state['query']}\n\nContext:\n{context}\n\nAnswer:"
	from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
	resp = llm.invoke([SystemMessage(content=prompt_system), HumanMessage(content=human_payload)])
	ai_message = AIMessage(content=resp.content.strip())
	state["answer"] = [ai_message]
	return {"answer": [ai_message]}