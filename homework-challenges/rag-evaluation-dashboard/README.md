# RAG Evaluation Dashboard — enhanced edition

My extended take on the Week 5 RAG evaluation idea. The course version (Ed Donner's) lives at
[`lectures/week-five/evaluator.py`](../../lectures/week-five/evaluator.py) and runs a fixed
evaluation over the test set. This version keeps that core idea and adds a set of features I built
to make the dashboard a real experimentation tool.

## What I added on top of the course version

- **🎛️ Tunable pipeline settings** — sliders for **chunk size**, **chunk overlap**, and **retrieval k**.
  Changing the chunk settings rebuilds an *in-memory* vector store (re-chunking + re-embedding the
  knowledge base) so I can compare ingestion strategies — without ever overwriting the saved `vector_db`.
- **🔀 Pipeline selector** — evaluate one of three answer pipelines:
  - **Built-in (tunable)** — the dashboard's own pipeline, driven by the sliders.
  - **Basic** — `implementation.answer` (the Day 1–4 LangChain pipeline).
  - **Pro** — `pro_implementation.answer` (the Day 5 advanced pipeline: query rewriting + dual
    retrieval + reranking).
- **📊 Modern, tabbed UI** with interactive **Plotly** charts (replacing the single bar chart).
- **🗂️ Per-category & per-knowledge-base breakdowns** — see which question types are hard and which
  KB sections (`company`, `contracts`, `employees`, `products`) actually supply the retrieved context.
- **📝 Human-readable summary reports** — after each run, an auto-generated narrative that interprets
  the numbers (with 🟢/🟠/🔴 verdicts), names the best/worst categories, and recommends a next step.
- **🔎 Lowest-scoring answers table** — the worst answers with the LLM judge's feedback, for debugging.

## What it measures

1. **Retrieval** — MRR, nDCG, keyword coverage (did we fetch the right context?).
2. **Answer quality** — LLM-as-a-judge accuracy / completeness / relevance (is the final answer good?).

## Run it

```bash
python homework-challenges/rag-evaluation-dashboard/evaluator.py
```

It automatically locates `lectures/week-five/` to read the knowledge base, vector store, and
`tests.jsonl` (no data is duplicated), and imports the basic/pro answer modules from there.

> Heads-up: the **Pro** pipeline and **answer-quality** runs call the LLM multiple times per
> question, so start with a small test count when iterating.
