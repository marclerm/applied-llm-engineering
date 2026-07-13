"""
loaders.py — download and curate a whole Amazon product category in parallel.

ItemLoader pulls one category from the McAuley-Lab/Amazon-Reviews-2023 dataset on
HuggingFace and runs `parser.parse` over every datapoint, fanning the work out
across CPU processes (each category has hundreds of thousands of rows).

⚠️ Adaptation note: Ed's original used `datasets.load_dataset(..., trust_remote_code=True)`.
That dataset ships a *loading script*, which `datasets >= 4` no longer supports
(and which also trips a pyarrow schema-cast error on the `images` field). So here we
read the raw `meta_<category>.jsonl` file directly from the Hub and normalize the two
fields `parse` cares about (`details` -> JSON string, `price` -> string). Same result,
robust across library versions.

Heads-up: this still downloads a large file per category and pins your CPU while it
runs. Lower `workers` (or pass workers=1) if it's too heavy on your machine.
"""

import os
import json
import gzip
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
from huggingface_hub import hf_hub_download

from pricer.parser import parse

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
CHUNK_SIZE = 1000  # how many datapoints each worker task handles at once

# Use all CPUs but one, so the machine stays responsive.
cpu_count = os.cpu_count()
WORKERS = max(cpu_count - 1, 1)


def _normalize(datapoint: dict) -> dict:
    """
    Make a raw jsonl record look like what `parse` expects.

    The loading script used to serialize `details` to a JSON string and cast
    `price` to a string; the raw jsonl has `details` as a dict and `price` as a
    string-or-None. We reproduce that here so parse() works unchanged.
    """
    details = datapoint.get("details")
    if isinstance(details, (dict, list)):
        datapoint["details"] = json.dumps(details)
    price = datapoint.get("price")
    datapoint["price"] = "" if price is None else str(price)
    return datapoint


def fetch_raw_dataset(category: str) -> list[dict]:
    """
    Download the raw metadata for a category from the Hub and return it as a list
    of datapoint dicts (one per product), normalized for `parse`.

    Reads `raw/meta_categories/meta_<category>.jsonl` directly — see the module
    docstring for why we avoid datasets.load_dataset here.
    """
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"raw/meta_categories/meta_{category}.jsonl",
        repo_type="dataset",
    )
    # Support both plain and gzipped files, just in case.
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [_normalize(json.loads(line)) for line in f]


def _process_chunk(args):
    """
    Worker: curate one chunk of datapoints into Items.

    This is a module-level function (not a bound method) on purpose: ProcessPoolExecutor
    pickles the callable for every task, so a bound method would copy the *entire*
    dataset to each worker. Here only the small chunk + category string travel.
    """
    chunk, category = args
    batch = [parse(datapoint, category) for datapoint in chunk]
    return [item for item in batch if item is not None]


class ItemLoader:
    def __init__(self, category):
        self.category = category
        self.dataset = None

    def chunk_generator(self):
        """Yield the dataset in CHUNK_SIZE slices, so workers each get a manageable batch."""
        size = len(self.dataset)
        for i in range(0, size, CHUNK_SIZE):
            yield self.dataset[i:i + CHUNK_SIZE]

    def load_in_parallel(self, workers):
        """
        Farm chunk-processing out to a process pool. Big speed-up over serial,
        but it ties up the machine while running.
        """
        results = []
        chunk_count = (len(self.dataset) // CHUNK_SIZE) + 1
        args = ((chunk, self.category) for chunk in self.chunk_generator())
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in tqdm(pool.map(_process_chunk, args), total=chunk_count):
                results.extend(batch)
        return results

    def load(self, workers=WORKERS):
        """Download this category from the Hub and curate it into a list of Items."""
        start = datetime.now()
        print(f"Loading dataset {self.category}", flush=True)
        self.dataset = fetch_raw_dataset(self.category)
        results = self.load_in_parallel(workers)
        finish = datetime.now()
        print(
            f"Completed {self.category} with {len(results):,} datapoints in "
            f"{(finish - start).total_seconds() / 60:.1f} mins",
            flush=True,
        )
        return results
