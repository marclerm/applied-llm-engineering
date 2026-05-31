"""
ingest.py (PRO) — Build the *advanced* RAG knowledge base.

This is the production-grade version of the basic `implementation/ingest.py`.
It's the "offline" / one-time step that builds the vector store, but with three
upgrades that come straight from Day 5's "Advanced RAG" notebook:

  1. LLM-driven chunking — instead of blindly cutting every N characters, an LLM
     splits each document into *sensible* overlapping chunks.
  2. Document pre-processing — for every chunk the LLM also writes a `headline`
     and a `summary`, which we store *alongside* the original text so retrieval
     has richer, more queryable signal to match against.
  3. Speed + resilience — documents are processed in parallel with a worker pool,
     and each LLM call is automatically retried with exponential backoff so a
     transient rate-limit doesn't kill the whole run.

Run it directly to (re)build the store:  `python ingest.py`

Adapted from ed-donner/llm_engineering (week5/pro_implementation/ingest.py).
"""

from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from multiprocessing import Pool
from tenacity import retry, wait_exponential

load_dotenv(override=True)

# The chat model used for chunking. litellm routes a bare OpenAI model name to
# OpenAI; prefix with a provider (e.g. "groq/...") to use a different backend.
MODEL = "gpt-4.1-nano"

# Paths resolve relative to THIS file, so the script works from any directory.
# parent.parent == lectures/week-five/
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")        # where Chroma persists
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"  # source Markdown docs

collection_name = "docs"
embedding_model = "text-embedding-3-large"  # MUST match the model used in answer.py

# Target average chunk size, in characters. We only use it to *suggest* a chunk
# count to the LLM. 100 is deliberately small for fine-grained, precise chunks —
# it produces many small chunks (more LLM output, but better retrieval precision).
# Raise it (e.g. 500) for fewer, larger chunks.
AVERAGE_CHUNK_SIZE = 100

# Retry policy for LLM calls: wait 10s, then back off exponentially up to 240s.
wait = wait_exponential(multiplier=1, min=10, max=240)

# How many documents to chunk in parallel. If you hit rate-limit errors, set to 1.
WORKERS = 3

openai = OpenAI()


class Result(BaseModel):
    """Our home-grown stand-in for LangChain's Document (content + metadata)."""

    page_content: str
    metadata: dict


class Chunk(BaseModel):
    """
    One chunk as produced by the LLM. The headline + summary are the
    "document pre-processing" payload: extra, highly-queryable text stored
    next to the verbatim original so a user's question is more likely to match.
    """

    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document):
        """Flatten this chunk into a Result, carrying the document's source/type metadata."""
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )


class Chunks(BaseModel):
    """Wrapper so the LLM can return a structured list of chunks via response_format."""

    chunks: list[Chunk]


def fetch_documents():
    """
    A homemade version of the LangChain DirectoryLoader.

    Walks each sub-folder of the knowledge base (company, contracts, employees,
    products), reads every .md file, and tags it with its folder name as `type`.
    """
    documents = []
    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append({"type": doc_type, "source": file.as_posix(), "text": f.read()})

    print(f"Loaded {len(documents)} documents")
    return documents


def make_prompt(document):
    """Build the chunking instruction for one document, suggesting a chunk count."""
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""


def make_messages(document):
    return [
        {"role": "user", "content": make_prompt(document)},
    ]


@retry(wait=wait)
def process_document(document):
    """
    Turn one document into a list of Result chunks via a single LLM call.

    Decorated with @retry so a transient error / rate-limit backs off and retries
    rather than aborting the whole ingest.
    """
    messages = make_messages(document)
    response = completion(model=MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]


def create_chunks(documents):
    """
    Chunk all documents in parallel using a pool of WORKERS processes.

    Parallelism matters here because chunking is one (slow) LLM call per document.
    If you get a rate-limit error, set WORKERS = 1 to fall back to serial.
    """
    chunks = []
    with Pool(processes=WORKERS) as pool:
        # imap_unordered streams results back as each worker finishes, so the
        # tqdm progress bar advances smoothly regardless of completion order.
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(result)
    return chunks


def create_embeddings(chunks):
    """
    Embed every chunk and store the vectors in a fresh Chroma collection.

    We delete any existing collection first so re-running gives a clean rebuild
    instead of appending duplicates.
    """
    chroma = PersistentClient(path=DB_NAME)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    # Embed the full page_content (headline + summary + original text) of each chunk.
    texts = [chunk.page_content for chunk in chunks]
    emb = openai.embeddings.create(model=embedding_model, input=texts).data
    vectors = [e.embedding for e in emb]

    collection = chroma.get_or_create_collection(collection_name)
    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")


if __name__ == "__main__":
    # The full offline pipeline: load -> LLM-chunk (parallel) -> embed/store.
    # The __main__ guard is required: on macOS the worker Pool re-imports this
    # module, and without the guard that would recursively spawn processes.
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
