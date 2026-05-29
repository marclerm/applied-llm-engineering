"""
ingest.py — Build the RAG knowledge base.

This is the "offline" / one-time step of a RAG pipeline. It:
  1. Reads every Markdown document in the knowledge-base folder.
  2. Splits each document into smaller overlapping chunks.
  3. Turns each chunk into an embedding vector.
  4. Stores those vectors in a persistent Chroma vector database.

Run it directly to (re)build the store:  `python ingest.py`

Adapted from ed-donner/llm_engineering (week5/implementation/ingest.py).
"""

import os
import glob
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: F401  (kept for the local-model option below)
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

# Chat model name — not used during ingestion, but kept here so all the
# project-wide constants live in one place.
MODEL = "gpt-4.1-nano"

# Paths are resolved relative to THIS file, so the script works no matter
# what directory you launch it from. parent.parent == lectures/week-five/
DB_NAME = str(Path(__file__).parent.parent / "vector_db")          # where Chroma persists its data
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")  # source Markdown documents

# Load environment variables (e.g. OPENAI_API_KEY) from a .env file.
# override=True lets the .env value win over anything already in the shell.
load_dotenv(override=True)

# The embedding model. This MUST be the same model used at query time
# (see answer.py) or Chroma raises a dimension-mismatch error.
#   - OpenAI text-embedding-3-large -> 3072 dimensions (used here)
#   - HuggingFace all-MiniLM-L6-v2  -> 384 dimensions (free/local alternative)
# To use the local model instead, comment the line below and uncomment:
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def fetch_documents():
    """
    Read the knowledge base from disk into LangChain Document objects.

    The knowledge base is organised as one sub-folder per category
    (company, contracts, employees, products). We loop over each folder,
    load every .md file inside it, and tag each document with a
    `doc_type` metadata field equal to the folder name. That metadata
    lets us later filter or colour-code documents by category.
    """
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)  # e.g. "products", "employees"
        # DirectoryLoader walks the folder; TextLoader reads each .md file as plain text.
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type  # remember which category this came from
            documents.append(doc)
    return documents


def create_chunks(documents):
    """
    Split whole documents into smaller, overlapping chunks.

    LLMs and retrieval work best on focused passages rather than entire
    documents. We use ~500-character chunks with 200 characters of overlap
    so that a sentence split across a boundary still appears (in full) in at
    least one chunk, preserving context for retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_embeddings(chunks):
    """
    Vectorize the chunks and store them in a persistent Chroma database.

    If a store already exists at DB_NAME we delete its collection first, so
    re-running this script gives a clean rebuild instead of appending
    duplicates. After building, we print the vector count and dimensionality
    as a quick sanity check.
    """
    # Wipe any previous collection so we don't accumulate stale/duplicate vectors.
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    # from_documents embeds every chunk and writes the vectors to disk in one call.
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    # Sanity check: report how many vectors were stored and their dimensionality.
    collection = vectorstore._collection
    count = collection.count()
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


if __name__ == "__main__":
    # The full offline pipeline: read -> chunk -> embed/store.
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
