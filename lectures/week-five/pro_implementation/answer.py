"""
answer.py (PRO) — Query the *advanced* RAG knowledge base.

This is the production-grade version of the basic `implementation/answer.py`.
It's the "online" / per-question step, layering several Advanced RAG techniques
from Day 5 on top of plain vector search:

  1. Query rewriting  — turn the user's (chatty) question into a sharp search query.
  2. Dual retrieval   — retrieve with BOTH the original and the rewritten query, then
                        merge/dedupe, so we get the best of both phrasings.
  3. Reranking        — an LLM re-orders the merged chunks by true relevance, and we
                        keep only the top FINAL_K for the final prompt.
  4. Resilience       — every LLM call is retried with exponential backoff.

It reads the store produced by this folder's `ingest.py`.

Key functions:
  - fetch_context(question)      -> the top reranked context chunks
  - answer_question(question, …) -> (answer text, context chunks)

Adapted from ed-donner/llm_engineering (week5/pro_implementation/answer.py).
"""

from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential

load_dotenv(override=True)

# litellm routes by provider prefix. Default to OpenAI's gpt-4.1-nano so this
# runs with just an OPENAI_API_KEY. To use a faster/cheaper open model via Groq,
# swap to the commented line below (requires a GROQ_API_KEY).
MODEL = "gpt-4.1-nano"
# MODEL = "groq/openai/gpt-oss-120b"

# Paths resolve relative to THIS file (parent.parent == lectures/week-five/).
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"

collection_name = "docs"
embedding_model = "text-embedding-3-large"  # MUST match the model used in ingest.py

# Retry policy: wait 10s, then back off exponentially up to 240s.
wait = wait_exponential(multiplier=1, min=10, max=240)

# RETRIEVAL_K: how many chunks to pull from Chroma per query (we over-fetch on
# purpose, since we run two queries and rerank). FINAL_K: how many survive the
# rerank and actually go into the answer prompt.
RETRIEVAL_K = 20
FINAL_K = 10

openai = OpenAI()

# Connect once at import time — the store was built by ingest.py.
chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(collection_name)

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""


class Result(BaseModel):
    """Our home-grown stand-in for LangChain's Document (content + metadata)."""

    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    """Structured output for the reranker: chunk ids, most relevant first."""

    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


@retry(wait=wait)
def rerank(question, chunks):
    """
    Ask an LLM to re-order the retrieved chunks by true relevance to the question.

    Vector similarity ≠ relevance, so we present the chunks (numbered) to the model
    and have it return the ids in best-first order. We then reorder accordingly.
    """
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        # Ids are 1-based here so the LLM never has to reason about a "chunk 0".
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(model=MODEL, messages=messages, response_format=RankOrder)
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    # Convert the 1-based ids back to 0-based indices to reorder our list.
    return [chunks[i - 1] for i in order]


def make_rag_messages(question, history, chunks):
    """Assemble the final chat messages: system prompt (with context) + history + question."""
    # Include each chunk's source file so the model can ground / attribute its answer.
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


@retry(wait=wait)
def rewrite_query(question, history=[]):
    """Rewrite the user's question into a short, specific Knowledge Base search query."""
    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a short, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    response = completion(model=MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content


def merge_chunks(chunks, reranked):
    """
    Combine two lists of chunks, dropping duplicates by page_content.

    Used to fuse the results of the original-query and rewritten-query searches.
    The first list is kept as-is; only genuinely new chunks from the second are added.
    """
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


def fetch_context_unranked(question):
    """Embed the question and pull the RETRIEVAL_K nearest chunks straight from Chroma."""
    query = openai.embeddings.create(model=embedding_model, input=[question]).data[0].embedding
    results = collection.query(query_embeddings=[query], n_results=RETRIEVAL_K)
    chunks = []
    for result in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=result[0], metadata=result[1]))
    return chunks

# The main event: the full advanced retrieval pipeline, returning the top FINAL_K chunks.
def fetch_context(original_question):
    """
    The full advanced retrieval pipeline:
      1. Rewrite the question into a sharper search query.
      2. Retrieve for BOTH the original and rewritten queries (dual retrieval).
      3. Merge + dedupe the two result sets.
      4. Rerank by relevance and keep the top FINAL_K.
    """
    rewritten_question = rewrite_query(original_question)
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    chunks = merge_chunks(chunks1, chunks2)
    reranked = rerank(original_question, chunks)
    return reranked[:FINAL_K]


@retry(wait=wait)
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    """Answer a question using advanced RAG; return the answer and the chunks that grounded it."""
    chunks = fetch_context(question)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks
