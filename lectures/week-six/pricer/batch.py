"""
batch.py — re-write every product into a clean, standard format using Groq's Batch API.

Day 2 of the project is *data pre-processing*: we ask an LLM to rewrite each messy,
scraped product description into a short, consistent summary (title / category / brand /
description / details). Doing this one-by-one would be slow and expensive, so we use the
**Batch API**, which trades latency (up to 24h) for a big discount and high throughput.

The `Batch` class wraps the whole dance:
  - split the items into groups of 1,000,
  - write each group as a `.jsonl` request file,
  - upload the file, submit a batch job, and track its id,
  - poll for completion, download the results, and write each summary back onto its Item,
  - (optionally) pickle the batch state so a long-running job survives a kernel restart.

We talk to Groq directly (not via litellm) because the Batch API is provider-specific.

Adapted from ed-donner/llm_engineering (week6/pricer/batch.py). Three changes worth noting:
  - Ed's file opens used `open(..., "rb", encoding="utf-8")`; binary mode doesn't accept an
    `encoding` argument (it raises ValueError on modern Python), so the encoding is dropped
    on the binary reads/writes here.
  - The Groq client is created lazily (on first use) rather than at import time, so you can
    import this module and run the non-Groq parts of the notebook without a GROQ_API_KEY set.
  - Added docstrings/comments throughout.
"""

import os
import json
import pickle
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv
from tqdm.notebook import tqdm

load_dotenv(override=True)

# Created lazily by _client() — see the module note. Day 2's batch operations need a
# GROQ_API_KEY in your .env (https://console.groq.com); only then is the client built.
_groq = None


def _client() -> Groq:
    """Return a cached Groq client, creating it on first use (needs GROQ_API_KEY)."""
    global _groq
    if _groq is None:
        _groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _groq


# A small, cheap open model is plenty for a mechanical rewrite-into-format task.
MODEL = "openai/gpt-oss-20b"
BATCHES_FOLDER = "batches"   # where we write the request .jsonl files
OUTPUT_FOLDER = "output"     # where we download the result .jsonl files
state = Path("batches.pkl")  # pickled Batch list, so a 24h job can survive a restart

# The model must answer in exactly this shape — no prose, no part numbers.
SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


class Batch:
    """One batch of (up to) 1,000 items submitted to Groq's Batch API as a single job."""

    BATCH_SIZE = 1_000

    # Class-level registry of all batches created in this session.
    batches = []

    def __init__(self, items, start, end, lite):
        self.items = items          # the full shared item list (sliced by start:end)
        self.start = start
        self.end = end
        self.filename = f"{start}_{end}.jsonl"
        self.file_id = None         # Groq file id after upload
        self.batch_id = None        # Groq batch job id after submit
        self.output_file_id = None  # Groq file id of the results, once complete
        self.done = False
        # Keep lite and full runs in separate folders so they don't clobber each other.
        folder = Path("lite") if lite else Path("full")
        self.batches = folder / BATCHES_FOLDER
        self.output = folder / OUTPUT_FOLDER
        self.batches.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)

    def make_jsonl(self, item):
        """Render one item as a single Batch-API request line (custom_id = item.id)."""
        body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item.full},
            ],
            "reasoning_effort": "low",
        }
        line = {
            "custom_id": str(item.id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        return json.dumps(line)

    def make_file(self):
        """Write this batch's slice of items to its request .jsonl file (one request per line)."""
        batch_file = self.batches / self.filename
        with batch_file.open("w", encoding="utf-8") as f:
            for item in self.items[self.start : self.end]:
                f.write(self.make_jsonl(item))
                f.write("\n")

    def send_file(self):
        """Upload the request file to Groq and remember its file id."""
        batch_file = self.batches / self.filename
        with batch_file.open("rb") as f:  # binary read — no encoding here (see module note)
            response = _client().files.create(file=f, purpose="batch")
        self.file_id = response.id

    def submit_batch(self):
        """Kick off the batch job for the uploaded file (24h completion window)."""
        response = _client().batches.create(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=self.file_id,
        )
        self.batch_id = response.id

    def is_ready(self):
        """Poll the job; if it's completed, stash the output file id and return True."""
        response = _client().batches.retrieve(self.batch_id)
        status = response.status
        if status == "completed":
            self.output_file_id = response.output_file_id
        return status == "completed"

    def fetch_output(self):
        """Download the completed results file to the output folder."""
        output_file = str(self.output / self.filename)
        response = _client().files.content(self.output_file_id)
        response.write_to_file(output_file)

    def apply_output(self):
        """Read each result line and write the generated summary back onto its Item."""
        output_file = str(self.output / self.filename)
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                json_line = json.loads(line)
                id = int(json_line["custom_id"])
                summary = json_line["response"]["body"]["choices"][0]["message"]["content"]
                self.items[id].summary = summary
        self.done = True

    @classmethod
    def create(cls, items, lite):
        """Slice the items into BATCH_SIZE groups and register a Batch for each."""
        for start in range(0, len(items), cls.BATCH_SIZE):
            end = min(start + cls.BATCH_SIZE, len(items))
            batch = Batch(items, start, end, lite)
            cls.batches.append(batch)
        print(f"Created {len(cls.batches)} batches")

    @classmethod
    def run(cls):
        """Write, upload and submit every registered batch."""
        for batch in tqdm(cls.batches):
            batch.make_file()
            batch.send_file()
            batch.submit_batch()
        print(f"Submitted {len(cls.batches)} batches")

    @classmethod
    def fetch(cls):
        """Check each unfinished batch; download + apply results for any that are ready."""
        for batch in tqdm(cls.batches):
            if not batch.done:
                if batch.is_ready():
                    batch.fetch_output()
                    batch.apply_output()
        finished = [batch for batch in cls.batches if batch.done]
        print(f"Finished {len(finished)} of {len(cls.batches)} batches")

    @classmethod
    def save(cls):
        """
        Pickle the batch list so a long-running job survives a kernel restart.

        We temporarily null out the (huge, shared) items list before pickling so the
        state file stays small, then restore it afterwards.
        """
        items = cls.batches[0].items
        for batch in cls.batches:
            batch.items = None
        with state.open("wb") as f:  # binary write — no encoding here (see module note)
            pickle.dump(cls.batches, f)
        for batch in cls.batches:
            batch.items = items
        print(f"Saved {len(cls.batches)} batches")

    @classmethod
    def load(cls, items):
        """Restore a pickled batch list and re-attach the shared items list to each batch."""
        with state.open("rb") as f:  # binary read — no encoding here (see module note)
            cls.batches = pickle.load(f)
        for batch in cls.batches:
            batch.items = items
        print(f"Loaded {len(cls.batches)} batches")
