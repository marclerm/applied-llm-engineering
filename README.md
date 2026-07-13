# Applied LLM Engineering

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude-D4A373)
![Google Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-local%20models-000000)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-FFD21E)
![LangChain](https://img.shields.io/badge/LangChain-1.0-1C3C3C)
![Chroma](https://img.shields.io/badge/Chroma-vector%20store-FF6B6B)
![Gradio](https://img.shields.io/badge/Gradio-UIs-FF7C00?logo=gradio&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-viz-3F4F75?logo=plotly&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-pipelines-2EA44F)

> **Tags:** `LLMs` · `Generative AI` · `Prompt Engineering` · `Streaming` · `Tokenization` ·
> `Frontier Models` · `Open-Source Models` · `Tool / Function Calling` · `Multimodal` ·
> `Hugging Face Transformers` · `Pipelines` · `Whisper / Audio` · `Code Generation` ·
> `RAG` · `Embeddings` · `Vector Stores` · `Reranking` · `Query Rewriting` ·
> `LLM-as-a-Judge` · `Evaluation (MRR / nDCG)` · `Gradio` · `Ollama` · `LangChain`

A personal, hands-on collection of LLM engineering experiments — built while working through
**"Become an LLM Engineer in 8 weeks: Build and deploy 8 LLM apps, mastering Generative AI,
RAG, LoRA and AI Agents"** by **Ed Donner** on Udemy.

## About this repo

1. 📚 **Course-based.** Everything here grows out of concepts learned in Ed Donner's Udemy course.
2. 🎯 **Purpose is learning.** It re-expresses the course material through similar examples, with my
   own custom experience layered on top, organized in a structure that's easier for *me* to follow.
3. 🔗 **Source material.** Most lectures are based on the companion repo Ed shares in the course,
   [ed-donner/llm_engineering](https://github.com/ed-donner/llm_engineering).
4. ✍️ **My own additions.** I've added my own comments throughout and expanded several of the
   samples/lectures with notes and ideas I found important while learning — written directly into
   the notebooks and modules.

> This is a study repo, not a product. Names like *Insurellm* and the knowledge-base data come from
> the course materials and are used purely for learning.

## What's inside, week by week

### 🗓️ Week 1 — Foundations & frontier APIs
`OpenAI chat completions` · `Ollama (local)` · `web scraping` · `tokenization` · `streaming` · `Markdown output`
- **1 — Chat completions & local Ollama:** first calls to a frontier model and a local model.
- **2 — Tokenizing & memory:** how tokenizers work and how conversation context is built up.
- **3 — Company brochure generator:** scrape a website (BeautifulSoup), summarize, and stream a
  Markdown brochure.

### 🗓️ Week 2 — Frontier models, UIs, tools & multimodal
`OpenAI` · `Anthropic Claude` · `Google Gemini` · `Gradio` · `function/tool calling` · `multimodal (image/audio)`
- **1 — Frontier model APIs:** comparing OpenAI, Anthropic and Google interfaces + streaming.
- **2 — Gradio LLM interfaces:** wrapping models in quick web UIs.
- **3 — Conversational AI chatbot:** stateful chat with history.
- **4 — Airline tool-calling assistant:** giving the LLM tools/functions to call.
- **5 — Multimodal airline assistant:** adding images and audio to the assistant.

### 🗓️ Week 3 — Open-source models & Hugging Face
`Hugging Face Transformers` · `pipelines` · `tokenizers` · `Whisper (audio)` · `token-probability visualization`
- **1 — Visualize token-by-token:** a graph visualization of how a model picks the next token,
  plus a **meeting-minutes creator** (audio transcription → structured minutes).
- `visualizer.py` — helper that builds the token-probability graph (networkx + matplotlib).

### 🗓️ Week 4 — Code generation
`frontier vs open-source code models` · `Gradio` · `C++ / Rust` · `running compiled output`
- **1 — Code generator:** generate optimized code from a prompt with a frontier model.
- **2 — Open-source code generator:** the same task with open-source models.
- **3 — Rust code generator:** targeting Rust, with model comparison.
- `styles.py`, `system_info.py` — UI styling and environment helpers.

### 🗓️ Week 5 — Retrieval-Augmented Generation (RAG)
`LangChain` · `Chroma` · `OpenAI / HuggingFace embeddings` · `t-SNE (Plotly)` · `evaluation` · `reranking` · `query rewriting`
- **1 — Expert knowledge worker (RAG):** intro to RAG over the Insurellm knowledge base.
- **2 — Vector store visualization:** build the Chroma store and explore it with 2D/3D t-SNE plots.
- **3 — RAG question-answering app:** a Gradio chat grounded in the vector store.
- **4 — RAG evaluation:** retrieval metrics (MRR, nDCG, keyword coverage) and **LLM-as-a-judge**
  answer scoring over 150 test questions.
- **5 — Advanced RAG techniques:** *native* (no-LangChain) ingestion with LLM-driven chunking and
  document pre-processing, plus **query rewriting** and **reranking**.
- **Supporting modules & apps:**
  - `implementation/` — the basic LangChain ingest + answer pipeline.
  - `pro_implementation/` — the production pipeline: parallel LLM chunking, retry/backoff, dual
    retrieval, merge + rerank.
  - `app.py` — a Gradio chat app on top of the RAG pipeline.
  - `evaluation/` + `evaluator.py` — the course **RAG evaluation dashboard** (retrieval metrics +
    LLM-as-a-judge) over the 150-question test set.

### 🗓️ Week 6 — "The Price is Right" capstone (data → baselines → deep learning & LLMs)
`Amazon Reviews dataset` · `data curation` · `LLM data pre-processing` · `Groq Batch API` · `litellm` · `traditional ML` · `scikit-learn` · `XGBoost` · `PyTorch` · `frontier LLMs (zero-shot)` · `evaluation harness`
A week-long capstone: predict a product's price from its description, trained on a scrape of
Amazon data. (Order of play: curation → pre-processing → baselines/traditional ML → deep
learning & LLMs → fine-tuning a frontier model.)
- **1 — Data curation:** download & scrub the McAuley Amazon-Reviews categories, build a
  price- and category-balanced sample, and push a curated dataset to the Hub.
- **2 — Data pre-processing:** rewrite every messy product into a clean, standard format with an
  LLM, run at scale through **Groq's Batch API**, and publish the pre-processed dataset.
- **3 — Evaluation, baselines & traditional ML:** a shared `evaluate` harness, trivial baselines
  (random / average), then linear regression (hand features + bag-of-words), Random Forest and
  XGBoost — all scored the same way for comparison.
- **4 — Neural networks & LLMs:** a "human" baseline, a from-scratch **PyTorch** neural network on
  bag-of-words features, then **frontier LLMs zero-shot** (GPT, Claude, Gemini, Grok via litellm) —
  no training, just world knowledge, scored on the same test set.
- **Supporting modules (`pricer/`):**
  - `items.py` — the `Item` data model (+ HuggingFace push/load helpers).
  - `parser.py` — `parse()`: clean one raw datapoint into an `Item` (or drop it).
  - `loaders.py` — `ItemLoader`: download & curate a whole category in parallel.
  - `batch.py` — the `Batch` class: split → upload → submit → poll → collect via the Groq Batch API.
  - `evaluator.py` — the shared `evaluate`/`Tester` harness (per-item error, MSE, r², charts).

### 🧪 Extras
- `homework-challenges/rag-evaluation-dashboard/` — **my own extended RAG evaluation dashboard**:
  tunable chunk size / overlap / k, a pipeline selector (built-in / basic / pro), per-category and
  per-knowledge-base breakdowns, and human-readable summary reports. (Builds on the course version
  in `lectures/week-five/`.)
- `homework-challenges/local-llms/` — an Ollama webpage-summarizer challenge.
- `projects/llama-weather-oracle/` — a small custom CLI project: live weather (Open-Meteo) +
  a local Llama response (Ollama) with an editable personality prompt.
- `lectures/utilities/scraper.py` — shared web-scraping helper.

## Repo structure

```
applied-llm-engineering/
├── lectures/
│   ├── week-one/      … foundations & frontier APIs
│   ├── week-two/      … UIs, tool calling, multimodal
│   ├── week-three/    … open-source models & Hugging Face
│   ├── week-four/     … code generation
│   ├── week-five/     … RAG (notebooks + implementation, pro_implementation, app, evaluator)
│   ├── week-six/      … "The Price is Right" capstone (notebooks + pricer/ package)
│   └── utilities/     … shared helpers
├── homework-challenges/
├── projects/
│   └── llama-weather-oracle/
├── requirements.txt
└── README.md
```

## Getting started

```bash
# 1. Create and activate a virtual environment (Python 3.14)
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the repo root with the keys for the services you want to use:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

For the local-model lectures, install [Ollama](https://ollama.com) and pull a model
(e.g. `ollama pull llama3.2`).

Then open any notebook with Jupyter / VS Code, or run the week-five apps directly:

```bash
python lectures/week-five/app.py         # RAG chat app
python lectures/week-five/evaluator.py   # RAG evaluation dashboard
```

> **Note:** generated vector stores (`vector_db/`, `preprocessed_db/`) and your `.env` are
> git-ignored. Build the stores by running the relevant week-five ingest step / notebook first.

## Acknowledgements

Huge thanks to **[Ed Donner](https://github.com/ed-donner)** for the excellent course and the
[ed-donner/llm_engineering](https://github.com/ed-donner/llm_engineering) repo that this work is
based on. All credit for the original course material and structure goes to him; the additions,
comments, reorganization and experiments here are my own learning notes.

## License

Released under the [MIT License](LICENSE) — the same license as Ed Donner's original
[ed-donner/llm_engineering](https://github.com/ed-donner/llm_engineering) repository. The original
course material and example data are © Ed Donner; this repository's additions and reorganization
are © Marco Lerma. The license retains Ed's copyright notice in full to credit his work.
