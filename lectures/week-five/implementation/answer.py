"""
answer.py — Query the RAG knowledge base.

This is the "online" / per-question step of a RAG pipeline. Given a user
question (and optional chat history) it:
  1. Retrieves the most relevant chunks from the Chroma store.
  2. Stuffs those chunks into the system prompt as context.
  3. Asks the LLM to answer using that context.

The vector store it reads is the one produced by ingest.py.

Key functions:
  - fetch_context(question)      -> the relevant context documents
  - answer_question(question, …) -> (answer text, context documents)

Adapted from ed-donner/llm_engineering (week5/implementation/answer.py).
"""

from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: F401  (kept for the local-model option below)
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from dotenv import load_dotenv

# Load environment variables (e.g. OPENAI_API_KEY) from a .env file.
load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
# Path to the store built by ingest.py. parent.parent == lectures/week-five/
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

# IMPORTANT: this embedding model must match the one used in ingest.py.
# Querying with a different model (different vector size) raises a Chroma
# "dimension mismatch" error.
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# How many chunks to retrieve per question. Higher = more context (and tokens).
RETRIEVAL_K = 10

# The system prompt. {context} is filled in at query time with the retrieved
# chunks, grounding the model's answer in our knowledge base.
SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

# Build the three long-lived objects once at import time so every question
# reuses them rather than re-loading the model/DB on each call:
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)  # connect to the store
retriever = vectorstore.as_retriever()                                          # similarity-search wrapper
llm = ChatOpenAI(temperature=0, model_name=MODEL)                               # temperature=0 -> deterministic


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve the chunks most relevant to `question`.

    Embeds the question with the same model used at ingest time, then asks
    Chroma for the RETRIEVAL_K nearest chunks by cosine similarity.
    """
    return retriever.invoke(question, k=RETRIEVAL_K)


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Merge prior user turns with the current question into one retrieval query.

    Pulling in earlier user messages helps retrieval handle follow-ups like
    "what about its price?" where the subject lives in the conversation, not
    the latest message. Note: only USER messages are included — assistant
    replies are skipped so we search on what the user actually asked.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer `question` with RAG. Returns (answer_text, context_documents).

    Steps:
      1. Build a retrieval query from history + question.
      2. Fetch the relevant chunks and join them into a context string.
      3. Assemble the message list: system prompt (with context),
         the prior conversation, then the new user question.
      4. Call the LLM and return its answer plus the documents used
         (the docs are handy for showing sources / citations in a UI).
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)

    # System prompt first, then replay the chat history, then the new question.
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))  # convert {role, content} dicts -> LangChain messages
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    return response.content, docs
