"""
RAG Evaluation Dashboard — enhanced edition (my homework-challenge build).

This is my extended take on the Week 5 RAG evaluation idea. It starts from the
concepts in Ed Donner's course (the simpler dashboard lives in
`lectures/week-five/evaluator.py`) and adds a lot of my own:

  - A modern, tabbed Gradio UI with interactive Plotly charts.
  - Tunable pipeline settings: chunk size / chunk overlap (rebuilds an in-memory
    store, leaving the saved vector_db untouched) and retrieval k.
  - A pipeline selector: evaluate the built-in tunable pipeline, the basic
    `implementation.answer`, or the pro `pro_implementation.answer`.
  - Per-question-category and per-knowledge-base-section breakdowns.
  - Human-readable summary reports that interpret the numbers and recommend a
    next step, plus a "lowest-scoring answers" table with the judge's feedback.

It evaluates on two fronts:
  1. Retrieval quality  - MRR, nDCG, keyword coverage (+ which KB sections we pull from)
  2. Answer quality      - LLM-as-a-judge: accuracy, completeness, relevance

Run it:

    python homework-challenges/rag-evaluation-dashboard/evaluator.py

It reads the knowledge base / vector store / tests from lectures/week-five/ (found
automatically), so no data is duplicated. Non-default chunk settings are built into
a temporary in-memory store so the saved vector_db is never overwritten.
"""

import os
import sys
import glob
import json
import math
import importlib
from collections import Counter, defaultdict
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Configuration (mirrors 4-rag-evaluation.ipynb / implementation/ingest.py)
# ---------------------------------------------------------------------------

MODEL = "gpt-4.1-nano"

# Default pipeline settings — these match how the persistent vector_db was built,
# so the dashboard starts instantly without re-embedding anything.
DEFAULT_K = 10
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 200

HERE = Path(__file__).parent

# This dashboard lives in homework-challenges/, but the data and answer modules
# live in lectures/week-five/. Walk up to find that folder so it works from
# anywhere, then read the knowledge base / vector store / tests from there and
# put it on sys.path so `implementation.answer` / `pro_implementation.answer` import.
for candidate in [HERE, *HERE.parents]:
    possible = candidate / "lectures" / "week-five"
    if (possible / "knowledge-base").exists():
        WEEK_FIVE = possible
        break
else:
    WEEK_FIVE = HERE

sys.path.insert(0, str(WEEK_FIVE))  # so `implementation.answer` / `pro_implementation.answer` import cleanly
DB_NAME = str(WEEK_FIVE / "vector_db")
KNOWLEDGE_BASE = str(WEEK_FIVE / "knowledge-base")
TESTS_FILE = str(WEEK_FIVE / "tests.jsonl")

# Question categories in tests.jsonl, in a sensible display order.
CATEGORY_ORDER = [
    "direct_fact", "temporal", "spanning", "comparative",
    "numerical", "relationship", "holistic",
]
# doc_type sections in the knowledge base.
KB_SECTIONS = ["company", "contracts", "employees", "products"]

SYSTEM_PROMPT_TEMPLATE = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(temperature=0, model_name=MODEL)


# ---------------------------------------------------------------------------
# Tunable pipeline: knowledge base → chunks → vector store → retriever
# ---------------------------------------------------------------------------

_DOCS: list[Document] | None = None        # cached knowledge-base documents
_STORE_CACHE: dict[tuple, Chroma] = {}     # cached vector stores keyed by (chunk_size, chunk_overlap)
RETRIEVER = None                           # the currently active retriever


def load_knowledge_base() -> list[Document]:
    """Read every Markdown document, tagging each with its `doc_type` (folder name). Cached."""
    global _DOCS
    if _DOCS is None:
        docs = []
        for folder in glob.glob(str(Path(KNOWLEDGE_BASE) / "*")):
            doc_type = os.path.basename(folder)
            loader = DirectoryLoader(
                folder, glob="**/*.md", loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
            )
            for doc in loader.load():
                doc.metadata["doc_type"] = doc_type
                docs.append(doc)
        _DOCS = docs
    return _DOCS


def get_store(chunk_size: int, chunk_overlap: int) -> Chroma:
    """
    Return a Chroma store for the given chunk settings.

    - Default settings reuse the persistent `vector_db` on disk (no re-embedding).
    - Any other settings build a temporary in-memory store (re-chunks + re-embeds the
      knowledge base) so the saved vector_db is never overwritten. Stores are cached.
    """
    key = (chunk_size, chunk_overlap)
    if key in _STORE_CACHE:
        return _STORE_CACHE[key]

    if key == (DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP) and os.path.exists(DB_NAME):
        store = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(load_knowledge_base())
        store = Chroma.from_documents(
            documents=chunks, embedding=embeddings,
            collection_name=f"eval_{chunk_size}_{chunk_overlap}",  # unique in-memory collection
        )
    _STORE_CACHE[key] = store
    return store


def configure_pipeline(chunk_size: int, chunk_overlap: int, k: int) -> None:
    """Point the active retriever at a store with the requested chunk settings and k."""
    global RETRIEVER
    RETRIEVER = get_store(chunk_size, chunk_overlap).as_retriever(search_kwargs={"k": k})


# Start on the defaults so importing / launching is instant.
configure_pipeline(DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_K)


def fetch_context(question: str) -> list[Document]:
    """Retrieve the relevant context documents for a question (uses the active retriever)."""
    return RETRIEVER.invoke(question)


def answer_question(question: str) -> tuple[str, list[Document]]:
    """Answer the question with RAG; return the answer AND the context documents."""
    docs = fetch_context(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
    return response.content, docs


# ---------------------------------------------------------------------------
# Pipeline selection: evaluate the built-in tunable pipeline, or one of the
# external answer modules (basic / pro). All three expose the same interface:
#   fetch_context(question)            -> list of chunks (each has .page_content + .metadata)
#   answer_question(question, history) -> (answer_text, chunks)
# ---------------------------------------------------------------------------

PIPELINE_BUILTIN = "Built-in (tunable)"
PIPELINE_BASIC = "Basic — implementation.answer"
PIPELINE_PRO = "Pro — pro_implementation.answer"
PIPELINE_CHOICES = [PIPELINE_BUILTIN, PIPELINE_BASIC, PIPELINE_PRO]

_MODULE_CACHE: dict[str, object] = {}


def _doc_type(metadata: dict) -> str:
    """Robustly read a chunk's section: basic uses 'doc_type', pro uses 'type'."""
    return metadata.get("doc_type") or metadata.get("type") or "unknown"


def _load_module(modname: str):
    """Import an external answer module on demand (and cache it)."""
    if modname not in _MODULE_CACHE:
        _MODULE_CACHE[modname] = importlib.import_module(modname)
    return _MODULE_CACHE[modname]


def _prepare_builtin(chunk_size, chunk_overlap, k, progress):
    """Validate settings and (re)build the built-in vector store if needed. Returns ints."""
    chunk_size, chunk_overlap, k = int(chunk_size), int(chunk_overlap), int(k)
    if chunk_overlap >= chunk_size:
        raise gr.Error("Chunk overlap must be smaller than chunk size.")
    needs_build = (chunk_size, chunk_overlap) not in _STORE_CACHE and not (
        (chunk_size, chunk_overlap) == (DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP) and os.path.exists(DB_NAME)
    )
    if needs_build:
        progress(0, desc=f"Rebuilding vector store (chunk {chunk_size}/{chunk_overlap}) — re-embedding…")
    configure_pipeline(chunk_size, chunk_overlap, k)
    return chunk_size, chunk_overlap, k


def get_pipeline(choice, chunk_size, chunk_overlap, k, progress):
    """
    Resolve the chosen pipeline to (fetch_fn, answer_fn, metric_k, settings_line).

    - Built-in: honours the chunk size / overlap / k sliders (rebuilding if needed).
    - Basic / Pro: import the external module; the sliders don't apply (those
      modules have their own fixed config), so metric_k comes from the module.
    """
    if choice == PIPELINE_BASIC:
        progress(0, desc="Loading basic pipeline (implementation.answer)…")
        mod = _load_module("implementation.answer")
        metric_k = getattr(mod, "RETRIEVAL_K", 10)
        return mod.fetch_context, mod.answer_question, metric_k, f"{PIPELINE_BASIC}, k={metric_k}"
    if choice == PIPELINE_PRO:
        progress(0, desc="Loading pro pipeline (pro_implementation.answer)…")
        mod = _load_module("pro_implementation.answer")
        metric_k = getattr(mod, "FINAL_K", 10)
        return mod.fetch_context, mod.answer_question, metric_k, f"{PIPELINE_PRO}, k={metric_k}"
    # Default: the built-in tunable pipeline.
    chunk_size, chunk_overlap, k = _prepare_builtin(chunk_size, chunk_overlap, k, progress)
    return fetch_context, answer_question, k, f"built-in · chunk {chunk_size}/{chunk_overlap}, k={k}"


# ---------------------------------------------------------------------------
# Test set
# ---------------------------------------------------------------------------


class TestQuestion(BaseModel):
    """A test question with expected keywords and a reference answer."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category (e.g., direct_fact, spanning, temporal)")


def load_tests() -> list[TestQuestion]:
    """Load test questions from the JSONL file."""
    tests = []
    with open(TESTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            tests.append(TestQuestion(**json.loads(line.strip())))
    return tests


# ---------------------------------------------------------------------------
# Retrieval metrics: MRR, nDCG, keyword coverage
# ---------------------------------------------------------------------------


class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr: float = Field(description="Mean Reciprocal Rank - averaged across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in the top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


def calculate_mrr(keyword: str, retrieved_docs: list[Document]) -> float:
    """Reciprocal rank for a single keyword: 1/rank of the first chunk that contains it."""
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    """Discounted Cumulative Gain."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list[Document], k: int) -> float:
    """nDCG for a single keyword using binary relevance (keyword present / absent)."""
    keyword_lower = keyword.lower()
    relevances = [1 if keyword_lower in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]]
    dcg = calculate_dcg(relevances, k)
    idcg = calculate_dcg(sorted(relevances, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def score_retrieval(test: TestQuestion, retrieved_docs: list[Document], k: int) -> RetrievalEval:
    """Score already-retrieved documents against a test's keywords."""
    mrr_scores = [calculate_mrr(kw, retrieved_docs) for kw in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    ndcg_scores = [calculate_ndcg(kw, retrieved_docs, k) for kw in test.keywords]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (keywords_found / total_keywords * 100) if total_keywords else 0.0

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


def evaluate_retrieval(test: TestQuestion, k: int = DEFAULT_K) -> RetrievalEval:
    """Evaluate retrieval performance for a single test question (built-in pipeline)."""
    return score_retrieval(test, fetch_context(test.question), k)


# ---------------------------------------------------------------------------
# Answer quality: LLM as a judge
# ---------------------------------------------------------------------------


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(description="Concise feedback comparing the answer to the reference answer")
    accuracy: float = Field(description="Factual correctness vs reference. 1 (wrong) to 5 (perfect). 3 is acceptable.")
    completeness: float = Field(description="Covers all aspects of the question. 1 (missing key info) to 5 (all reference info included).")
    relevance: float = Field(description="Directly addresses the question with no extra info. 1 (off-topic) to 5 (perfectly relevant).")


judge = ChatOpenAI(temperature=0, model_name=MODEL).with_structured_output(AnswerEval)

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator assessing the quality of answers. "
    "Evaluate the generated answer by comparing it to the reference answer. "
    "Only give 5/5 scores for perfect answers."
)


def evaluate_answer(test: TestQuestion, answer_fn=answer_question) -> tuple[AnswerEval, str, list]:
    """Evaluate answer quality using LLM-as-a-judge, against the given answer pipeline."""
    generated_answer, retrieved_docs = answer_fn(test.question)

    judge_prompt = f"""Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 for perfect answers. If the answer is wrong, accuracy must be 1.
2. Completeness: How thoroughly does it cover all the information from the reference answer?
3. Relevance: How directly does it answer the question, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension."""

    answer_eval = judge.invoke([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=judge_prompt),
    ])
    return answer_eval, generated_answer, retrieved_docs


def evaluate_all_retrieval(limit: int | None, k: int, fetch_fn):
    """Evaluate retrieval per test with `fetch_fn`, yielding (test, result, doc_types, progress)."""
    tests = load_tests()
    if limit:
        tests = tests[:limit]
    total = len(tests)
    for index, test in enumerate(tests):
        docs = fetch_fn(test.question)
        result = score_retrieval(test, docs, k)
        doc_types = [_doc_type(d.metadata) for d in docs]
        yield test, result, doc_types, (index + 1) / total


def evaluate_all_answers(limit: int | None, answer_fn):
    """Evaluate answers per test with `answer_fn`, yielding (test, result, generated_answer, progress)."""
    tests = load_tests()
    if limit:
        tests = tests[:limit]
    total = len(tests)
    for index, test in enumerate(tests):
        result, generated_answer, _ = evaluate_answer(test, answer_fn)
        yield test, result, generated_answer, (index + 1) / total


# ---------------------------------------------------------------------------
# Dashboard helpers: color coding, metric cards, interpretation
# ---------------------------------------------------------------------------

# Color coding thresholds - Retrieval
MRR_GREEN, MRR_AMBER = 0.9, 0.75
NDCG_GREEN, NDCG_AMBER = 0.9, 0.75
COVERAGE_GREEN, COVERAGE_AMBER = 90.0, 75.0
# Color coding thresholds - Answer (1-5 scale)
ANSWER_GREEN, ANSWER_AMBER = 4.5, 4.0

THRESHOLDS = {
    "mrr": (MRR_GREEN, MRR_AMBER),
    "ndcg": (NDCG_GREEN, NDCG_AMBER),
    "coverage": (COVERAGE_GREEN, COVERAGE_AMBER),
    "accuracy": (ANSWER_GREEN, ANSWER_AMBER),
    "completeness": (ANSWER_GREEN, ANSWER_AMBER),
    "relevance": (ANSWER_GREEN, ANSWER_AMBER),
}
COLORS = {"good": "#1a9850", "ok": "#f0883e", "poor": "#d73027"}


def grade(value: float, metric_type: str) -> str:
    """Return 'good' / 'ok' / 'poor' for a metric value."""
    green, amber = THRESHOLDS[metric_type]
    if value >= green:
        return "good"
    if value >= amber:
        return "ok"
    return "poor"


def metric_card(label: str, value: float, metric_type: str, suffix: str) -> str:
    """A single color-coded metric card."""
    color = COLORS[grade(value, metric_type)]
    value_str = f"{value:.1f}%" if suffix == "%" else f"{value:.2f}{suffix}"
    return f"""
    <div style="flex: 1; min-width: 160px; margin: 6px; padding: 18px; background: var(--block-background-fill);
                border-radius: 12px; border: 1px solid var(--border-color-primary); border-top: 5px solid {color};
                box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
        <div style="font-size: 13px; color: var(--body-text-color-subdued); margin-bottom: 6px;
                    text-transform: uppercase; letter-spacing: .04em;">{label}</div>
        <div style="font-size: 32px; font-weight: 700; color: {color};">{value_str}</div>
    </div>
    """


def cards_row(cards: list[str], count: int, settings_note: str) -> str:
    """Wrap metric cards in a flex row with a completion banner showing the settings used."""
    return f"""
    <div style="display: flex; flex-wrap: wrap; gap: 4px;">{''.join(cards)}</div>
    <div style="margin-top: 14px; padding: 10px; background: rgba(26,152,80,0.12); border-radius: 8px;
                text-align: center; color: var(--body-text-color);">
        ✓ Evaluation complete — {count} test{'s' if count != 1 else ''} · {settings_note}
    </div>
    """


_VERDICT = {"good": "strong ✅", "ok": "acceptable ⚠️", "poor": "needs work ❌"}


def _overall_verdict(grades: list[str], subject: str) -> str:
    """One-line headline based on the mix of per-metric grades."""
    if all(g == "good" for g in grades):
        return f"🟢 **Overall: {subject} is in great shape.**"
    if any(g == "poor" for g in grades):
        return "🔴 **Overall: there's a real problem to fix here.**"
    return "🟠 **Overall: acceptable, but with clear room to improve.**"


def _pretty(name: str) -> str:
    """direct_fact -> 'direct fact'."""
    return name.replace("_", " ")


def summarize_retrieval(avg, cat_summary, doc_type_counts, count, k, settings_line) -> str:
    """A human-readable narrative interpreting the retrieval run."""
    g_cov, g_mrr, g_ndcg = grade(avg["coverage"], "coverage"), grade(avg["mrr"], "mrr"), grade(avg["ndcg"], "ndcg")
    headline = _overall_verdict([g_cov, g_mrr, g_ndcg], "retrieval")

    by_mrr = sorted(cat_summary.items(), key=lambda kv: kv[1]["MRR"])
    worst_cat, worst_v = by_mrr[0]
    best_cat, best_v = by_mrr[-1]
    if len(by_mrr) == 1:
        cat_block = [f"- This slice only contained **{_pretty(best_cat)}** questions "
                     f"(MRR {best_v['MRR']:.2f}); widen the test count to compare categories."]
    else:
        cat_block = [
            f"- 💪 Best handled: **{_pretty(best_cat)}** questions (MRR {best_v['MRR']:.2f}).",
            f"- 😰 Hardest: **{_pretty(worst_cat)}** questions (MRR {worst_v['MRR']:.2f}) — multi-document "
            "question types like `spanning` and `holistic` are usually the toughest to retrieve for.",
        ]

    total = sum(doc_type_counts.values()) or 1
    top_sec, top_n = doc_type_counts.most_common(1)[0]
    missing = [s for s in KB_SECTIONS if doc_type_counts.get(s, 0) == 0]

    if g_cov == "poor":
        rec = ("**Coverage is the bottleneck** — for many questions the right documents simply aren't "
               f"being retrieved. Try a larger **retrieval k** (currently {k}), different **chunking**, "
               "or switch the **RAG pipeline** above (e.g. the Pro pipeline rewrites the query and reranks).")
    elif g_mrr != "good" and g_cov != "poor":
        rec = ("The right content **is** retrieved but tends to sit **lower in the ranking** (coverage is "
               "healthier than MRR). Smaller, more focused chunks or a re-ranker would help.")
    else:
        rec = ("Retrieval is solid across the board — your effort is better spent on the **Answer Quality** "
               "tab now.")

    lines = [
        "## 📝 Retrieval summary",
        f"_{count} test question{'s' if count != 1 else ''} · {settings_line}._",
        "",
        headline,
        "",
        "**What the numbers say**",
        f"- On average **{avg['coverage']:.0f}% of expected keywords** were found in the top {k} chunks "
        f"({_VERDICT[g_cov]}). This tells us whether the right documents are retrieved *at all*.",
        f"- **MRR is {avg['mrr']:.2f}** ({_VERDICT[g_mrr]}) — the first relevant chunk shows up around "
        f"position **{(1 / avg['mrr']):.1f}** on average, so the right content lands "
        f"{'near the top' if g_mrr == 'good' else 'but is often buried lower down'}.",
        f"- **nDCG is {avg['ndcg']:.2f}** ({_VERDICT[g_ndcg]}), confirming the overall ranking quality.",
        "",
        "**Where it's hard vs. easy**",
        *cat_block,
        "",
        "**Knowledge base**",
        f"- Most retrieved context came from the **{top_sec}** section "
        f"({top_n} chunks, {top_n / total * 100:.0f}% of all retrieved).",
        (f"- ⚠️ No chunks were ever pulled from: **{', '.join(missing)}** — worth checking whether those "
         "questions are covered." if missing else
         "- All four knowledge-base sections contributed context. 👍"),
        "",
        f"**👉 Recommendation:** {rec}",
    ]
    return "\n".join(lines)


def summarize_answer(avg, cat_summary, count, k, settings_line) -> str:
    """A human-readable narrative interpreting the answer-quality run."""
    g_acc = grade(avg["accuracy"], "accuracy")
    g_comp = grade(avg["completeness"], "completeness")
    g_rel = grade(avg["relevance"], "relevance")
    headline = _overall_verdict([g_acc, g_comp, g_rel], "answer quality")

    dims = {"Accuracy": avg["accuracy"], "Completeness": avg["completeness"], "Relevance": avg["relevance"]}
    weakest_dim = min(dims, key=dims.get)

    by_acc = sorted(cat_summary.items(), key=lambda kv: kv[1]["Accuracy"])
    worst_cat, worst_v = by_acc[0]
    best_cat, best_v = by_acc[-1]
    if len(by_acc) == 1:
        cat_block = [f"- This slice only contained **{_pretty(best_cat)}** questions "
                     f"(accuracy {best_v['Accuracy']:.1f}/5); widen the test count to compare categories."]
    else:
        cat_block = [
            f"- 💪 Most accurate on: **{_pretty(best_cat)}** questions ({best_v['Accuracy']:.1f}/5).",
            f"- 😰 Least accurate on: **{_pretty(worst_cat)}** questions ({worst_v['Accuracy']:.1f}/5).",
        ]

    rec_by_dim = {
        "Accuracy": ("Wrong facts are creeping in. This is usually a **retrieval** problem (the answer "
                     "isn't grounded in the right context) or the model hallucinating — check the Retrieval tab."),
        "Completeness": ("Answers are correct but **leave things out**. The reference info often isn't all "
                         f"being retrieved — try a larger **retrieval k** (currently {k}) or different chunking."),
        "Relevance": ("Answers drift or add **extra, unasked-for information**. Tightening the system prompt "
                      "to be more concise and on-topic should help."),
    }

    lines = [
        "## 📝 Answer-quality summary",
        f"_LLM-as-a-judge over {count} test question{'s' if count != 1 else ''} (scale 1–5) · {settings_line}._",
        "",
        headline,
        "",
        "**What the numbers say**",
        f"- **Accuracy {avg['accuracy']:.1f}/5** ({_VERDICT[g_acc]}) — the headline metric: are the facts "
        "right vs. the reference answer? Any wrong answer is scored 1, so this is unforgiving.",
        f"- **Completeness {avg['completeness']:.1f}/5** ({_VERDICT[g_comp]}) — did the answer cover "
        "*everything* it should have?",
        f"- **Relevance {avg['relevance']:.1f}/5** ({_VERDICT[g_rel]}) — did it answer directly without "
        "padding?",
        "",
        f"The weakest dimension is **{weakest_dim}** ({dims[weakest_dim]:.1f}/5).",
        "",
        "**Where it's hard vs. easy**",
        *cat_block,
        "",
        f"**👉 Recommendation:** {rec_by_dim[weakest_dim]}",
        "",
        "_See the lowest-scoring answers table below for concrete failure cases and the judge's feedback._",
    ]
    return "\n".join(lines)


def category_bar(rows, value_cols, title, y_range):
    """Grouped bar chart of metrics by question category."""
    df = pd.DataFrame(rows)
    long_df = df.melt(id_vars="Category", value_vars=value_cols, var_name="Metric", value_name="Score")
    long_df["Category"] = pd.Categorical(long_df["Category"], categories=CATEGORY_ORDER, ordered=True)
    long_df = long_df.sort_values("Category")
    fig = px.bar(
        long_df, x="Category", y="Score", color="Metric", barmode="group",
        title=title, range_y=y_range, template="plotly_white",
    )
    fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), legend_title_text="")
    return fig


def kb_bar(doc_type_counts: Counter):
    """Bar chart of which knowledge-base sections supplied the retrieved chunks."""
    total = sum(doc_type_counts.values()) or 1
    rows = [
        {"Section": s, "Chunks retrieved": doc_type_counts.get(s, 0),
         "Share": doc_type_counts.get(s, 0) / total * 100}
        for s in KB_SECTIONS
    ]
    for s, c in doc_type_counts.items():  # include any unexpected doc_types too
        if s not in KB_SECTIONS:
            rows.append({"Section": s, "Chunks retrieved": c, "Share": c / total * 100})
    df = pd.DataFrame(rows)
    fig = px.bar(
        df, x="Section", y="Chunks retrieved", color="Section", text="Chunks retrieved",
        title="Knowledge-base sections supplying retrieved context", template="plotly_white",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Dashboard callbacks
# ---------------------------------------------------------------------------


def run_retrieval_evaluation(pipeline_choice, chunk_size, chunk_overlap, k, num_tests, progress=gr.Progress()):
    """Run retrieval evaluation with the chosen pipeline + settings over the first N tests."""
    fetch_fn, _, metric_k, settings_line = get_pipeline(pipeline_choice, chunk_size, chunk_overlap, k, progress)
    limit = int(num_tests)
    totals = {"mrr": 0.0, "ndcg": 0.0, "coverage": 0.0}
    per_cat = defaultdict(lambda: defaultdict(list))
    doc_type_counts = Counter()
    count = 0

    for test, result, doc_types, prog in evaluate_all_retrieval(limit, metric_k, fetch_fn):
        count += 1
        totals["mrr"] += result.mrr
        totals["ndcg"] += result.ndcg
        totals["coverage"] += result.keyword_coverage
        per_cat[test.category]["MRR"].append(result.mrr)
        per_cat[test.category]["nDCG"].append(result.ndcg)
        per_cat[test.category]["Coverage %"].append(result.keyword_coverage)
        doc_type_counts.update(doc_types)
        progress(prog, desc=f"Evaluating test {count}/{limit}...")

    avg = {key: value / count for key, value in totals.items()}
    cards = [
        metric_card("Keyword Coverage", avg["coverage"], "coverage", "%"),
        metric_card("MRR", avg["mrr"], "mrr", ""),
        metric_card("nDCG", avg["ndcg"], "ndcg", ""),
    ]
    summary = cards_row(cards, count, settings_line)

    cat_summary = {
        cat: {"MRR": sum(m["MRR"]) / len(m["MRR"]),
              "nDCG": sum(m["nDCG"]) / len(m["nDCG"]),
              "Coverage %": sum(m["Coverage %"]) / len(m["Coverage %"])}
        for cat, m in per_cat.items()
    }
    cat_rows = [
        {"Category": cat, "MRR": v["MRR"], "nDCG": v["nDCG"], "Coverage %": v["Coverage %"] / 100}
        for cat, v in cat_summary.items()
    ]
    cat_fig = category_bar(cat_rows, ["MRR", "nDCG", "Coverage %"],
                           "Retrieval metrics by question category (0–1)", [0, 1])
    kb_fig = kb_bar(doc_type_counts)
    notes = summarize_retrieval(avg, cat_summary, doc_type_counts, count, metric_k, settings_line)
    return summary, cat_fig, kb_fig, notes


def run_answer_evaluation(pipeline_choice, chunk_size, chunk_overlap, k, num_tests, progress=gr.Progress()):
    """Run answer-quality evaluation with the chosen pipeline + settings over the first N tests."""
    _, answer_fn, metric_k, settings_line = get_pipeline(pipeline_choice, chunk_size, chunk_overlap, k, progress)
    limit = int(num_tests)
    totals = {"accuracy": 0.0, "completeness": 0.0, "relevance": 0.0}
    per_cat = defaultdict(lambda: defaultdict(list))
    worst = []
    count = 0

    for test, result, answer, prog in evaluate_all_answers(limit, answer_fn):
        count += 1
        totals["accuracy"] += result.accuracy
        totals["completeness"] += result.completeness
        totals["relevance"] += result.relevance
        per_cat[test.category]["Accuracy"].append(result.accuracy)
        per_cat[test.category]["Completeness"].append(result.completeness)
        per_cat[test.category]["Relevance"].append(result.relevance)
        worst.append({
            "Category": test.category,
            "Question": test.question,
            "Acc": result.accuracy,
            "Comp": result.completeness,
            "Rel": result.relevance,
            "Judge feedback": result.feedback,
        })
        progress(prog, desc=f"Evaluating test {count}/{limit}...")

    avg = {key: value / count for key, value in totals.items()}
    cards = [
        metric_card("Accuracy", avg["accuracy"], "accuracy", "/5"),
        metric_card("Completeness", avg["completeness"], "completeness", "/5"),
        metric_card("Relevance", avg["relevance"], "relevance", "/5"),
    ]
    summary = cards_row(cards, count, settings_line)

    cat_summary = {
        cat: {"Accuracy": sum(m["Accuracy"]) / len(m["Accuracy"]),
              "Completeness": sum(m["Completeness"]) / len(m["Completeness"]),
              "Relevance": sum(m["Relevance"]) / len(m["Relevance"])}
        for cat, m in per_cat.items()
    }
    cat_rows = [{"Category": cat, **v} for cat, v in cat_summary.items()]
    cat_fig = category_bar(cat_rows, ["Accuracy", "Completeness", "Relevance"],
                           "Answer quality by question category (1–5)", [1, 5])

    worst_df = pd.DataFrame(sorted(worst, key=lambda r: (r["Acc"], r["Comp"], r["Rel"]))[:10])
    notes = summarize_answer(avg, cat_summary, count, metric_k, settings_line)
    return summary, cat_fig, worst_df, notes


# ---------------------------------------------------------------------------
# Static explanatory copy
# ---------------------------------------------------------------------------

PLACEHOLDER = "<div style='padding: 24px; text-align:center; color: var(--body-text-color-subdued);'>Adjust the pipeline settings, set the number of tests, and click <b>Run</b>.</div>"

HOW_TO_READ = """
## How to read these metrics

Evaluation turns RAG from guesswork into engineering: with these numbers you can answer the
only question that matters when you change something — *did it get better or worse?*

### ⚙️ Pipeline settings you can tune
- **RAG pipeline** — *which* answer pipeline to evaluate:
  - **Built-in (tunable)** — the dashboard's own pipeline, configured by the sliders below.
  - **Basic (`implementation.answer`)** — the Day 1–4 LangChain pipeline reading `vector_db`.
  - **Pro (`pro_implementation.answer`)** — the Day 5 advanced pipeline reading `preprocessed_db`:
    query rewriting + dual retrieval + reranking. It's slower and costs more per question (several
    LLM calls each), so use a small test count first. The sliders below don't apply to Basic/Pro —
    those modules carry their own fixed configuration.
- **Chunk size** — characters per chunk when documents are split. Smaller = more focused but more
  fragmented; larger = more context per chunk but noisier retrieval.
- **Chunk overlap** — characters shared between consecutive chunks, so a fact split across a boundary
  still appears whole in at least one chunk. Must be smaller than chunk size.
- **Retrieval k** — how many chunks are fetched per query. More k = higher recall, but more noise in
  the prompt. Changing **k** is instant; changing **chunk size/overlap** rebuilds the vector store
  (re-embeds the whole knowledge base, into a temporary in-memory store — your saved `vector_db` is
  left untouched).

### 🔍 Retrieval — "did we fetch the right context?"
We use each test's **keywords** as a stand-in for "relevant", and check the top-k retrieved chunks.

| Metric | What it measures | 🟢 good | 🟠 ok | 🔴 poor |
|---|---|---|---|---|
| **Keyword coverage** | % of keywords found *anywhere* in the top-k | ≥ 90% | ≥ 75% | < 75% |
| **MRR** | How *early* the right chunk appears (1/rank, averaged) | ≥ 0.90 | ≥ 0.75 | < 0.75 |
| **nDCG** | Rank quality with a logarithmic discount | ≥ 0.90 | ≥ 0.75 | < 0.75 |

Coverage low → the documents aren't being retrieved at all (try a bigger `k`, better chunking,
or a different embedding model). Coverage high but MRR low → the content is there but ranked too low.

### 💬 Answer quality — "is the final answer good?" (LLM-as-a-judge, 1–5)
A second LLM compares our generated answer to the reference answer.

| Metric | What it measures | 🟢 good | 🟠 ok | 🔴 poor |
|---|---|---|---|---|
| **Accuracy** | Factually correct vs. reference (wrong ⇒ 1) | ≥ 4.5 | ≥ 4.0 | < 4.0 |
| **Completeness** | Covers everything in the reference | ≥ 4.5 | ≥ 4.0 | < 4.0 |
| **Relevance** | Answers directly, no rambling | ≥ 4.5 | ≥ 4.0 | < 4.0 |

### 📚 By category & knowledge base
- **By question category** breaks scores out across the 7 question types (`direct_fact`, `temporal`,
  `spanning`, …) so you can see *which kinds of questions* are hard — `spanning` and `holistic`
  questions that need many documents are usually the toughest.
- **Knowledge-base sections** shows which parts of the corpus (`employees`, `products`, `contracts`,
  `company`) supply the retrieved context — a quick sanity check that retrieval isn't ignoring a section.
"""


def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="emerald", font=["Inter", "system-ui", "sans-serif"])
    n_tests = len(load_tests())

    with gr.Blocks(title="RAG Evaluation Dashboard", theme=theme) as app:
        gr.Markdown("# 📊 RAG Evaluation Dashboard")
        gr.Markdown(
            "Evaluate **retrieval** and **answer quality** for the Insurellm RAG system, and tune the "
            f"pipeline live. Each run scores the first *N* of the **{n_tests}** test questions in "
            "`tests.jsonl`. _Tip: answer evaluation calls the LLM judge once per test — start small._"
        )

        # --- Shared pipeline settings -----------------------------------------
        with gr.Group():
            gr.Markdown("### ⚙️ Pipeline settings")
            pipeline = gr.Dropdown(
                PIPELINE_CHOICES, value=PIPELINE_BUILTIN, label="RAG pipeline to evaluate",
                info=("Built-in: the tunable pipeline configured by the sliders below. "
                      "Basic: implementation.answer. Pro: pro_implementation.answer "
                      "(query rewriting + dual retrieval + reranking — slower/costlier per question)."),
            )
            with gr.Row():
                chunk_size = gr.Slider(
                    100, 2000, value=DEFAULT_CHUNK_SIZE, step=50, label="Chunk size",
                    info="Characters per chunk when splitting documents. Rebuilds the vector store.",
                )
                chunk_overlap = gr.Slider(
                    0, 500, value=DEFAULT_CHUNK_OVERLAP, step=25, label="Chunk overlap",
                    info="Characters shared between neighbouring chunks (must be < chunk size). Rebuilds the store.",
                )
                k_slider = gr.Slider(
                    1, 20, value=DEFAULT_K, step=1, label="Retrieval k",
                    info="How many chunks to retrieve per query. Free to change — no rebuild.",
                )
            settings_caption = gr.Markdown(
                "_Changing chunk size/overlap re-embeds the knowledge base into a temporary in-memory "
                "store (takes a moment, uses OpenAI embeddings). Your saved `vector_db` is never overwritten._"
            )

        # The sliders only apply to the built-in pipeline; grey them out otherwise.
        def _toggle_sliders(choice):
            builtin = choice == PIPELINE_BUILTIN
            cap = (
                "_Changing chunk size/overlap re-embeds the knowledge base into a temporary in-memory "
                "store (takes a moment, uses OpenAI embeddings). Your saved `vector_db` is never overwritten._"
                if builtin else
                f"_The **{choice}** pipeline uses its own fixed configuration — the sliders above don't apply._"
            )
            return (
                gr.update(interactive=builtin),
                gr.update(interactive=builtin),
                gr.update(interactive=builtin),
                gr.update(value=cap),
            )

        pipeline.change(
            _toggle_sliders, inputs=pipeline,
            outputs=[chunk_size, chunk_overlap, k_slider, settings_caption],
        )

        with gr.Tabs():
            # --- Retrieval tab -------------------------------------------------
            with gr.Tab("🔍 Retrieval"):
                with gr.Row():
                    r_slider = gr.Slider(5, n_tests, value=n_tests, step=5,
                                         label="Number of tests to evaluate", scale=4)
                    r_button = gr.Button("Run retrieval evaluation", variant="primary", scale=1)
                r_summary = gr.HTML(PLACEHOLDER)
                with gr.Row():
                    r_cat_plot = gr.Plot(label="By question category")
                    r_kb_plot = gr.Plot(label="By knowledge-base section")
                r_notes = gr.Markdown()

            # --- Answer tab ----------------------------------------------------
            with gr.Tab("💬 Answer Quality"):
                with gr.Row():
                    a_slider = gr.Slider(5, n_tests, value=n_tests, step=5,
                                         label="Number of tests to evaluate", scale=4)
                    a_button = gr.Button("Run answer evaluation", variant="primary", scale=1)
                a_summary = gr.HTML(PLACEHOLDER)
                a_cat_plot = gr.Plot(label="By question category")
                a_notes = gr.Markdown()
                gr.Markdown("#### 🔎 Lowest-scoring answers (sorted by accuracy)")
                a_table = gr.Dataframe(
                    headers=["Category", "Question", "Acc", "Comp", "Rel", "Judge feedback"],
                    wrap=True, label=None,
                )

            # --- Help tab ------------------------------------------------------
            with gr.Tab("ℹ️ How to read this"):
                gr.Markdown(HOW_TO_READ)

        r_button.click(
            run_retrieval_evaluation,
            inputs=[pipeline, chunk_size, chunk_overlap, k_slider, r_slider],
            outputs=[r_summary, r_cat_plot, r_kb_plot, r_notes],
        )
        a_button.click(
            run_answer_evaluation,
            inputs=[pipeline, chunk_size, chunk_overlap, k_slider, a_slider],
            outputs=[a_summary, a_cat_plot, a_table, a_notes],
        )

    return app


def main():
    """Launch the Gradio evaluation app."""
    build_app().launch(inbrowser=True)


if __name__ == "__main__":
    main()
