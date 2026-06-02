"""
loaders.py — download and curate a whole Amazon product category in parallel.

ItemLoader pulls one category from the McAuley-Lab/Amazon-Reviews-2023 dataset
on HuggingFace and runs `parser.parse` over every datapoint, fanning the work
out across CPU processes (each category has hundreds of thousands of rows).

Heads-up: this downloads a large dataset and pins your CPU while it runs. Lower
`workers` (or pass workers=1) if it's too heavy on your machine.

Adapted from ed-donner/llm_engineering (week6/pricer/loaders.py).
"""

from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor
from pricer.parser import parse
import os

CHUNK_SIZE = 1000  # how many datapoints each worker task handles at once

# Use all CPUs but one, so the machine stays responsive.
cpu_count = os.cpu_count()
WORKERS = max(cpu_count - 1, 1)


class ItemLoader:
    def __init__(self, category):
        self.category = category
        self.dataset = None

    def from_datapoint(self, datapoint):
        """Try to create an Item from this datapoint; None if it shouldn't be included."""
        return parse(datapoint, self.category)

    def from_chunk(self, chunk):
        """Create a list of Items from a chunk of datapoints (dropping the None's)."""
        batch = [self.from_datapoint(datapoint) for datapoint in chunk]
        return [item for item in batch if item is not None]

    def chunk_generator(self):
        """Yield the dataset in CHUNK_SIZE slices, so workers each get a manageable batch."""
        size = len(self.dataset)
        for i in range(0, size, CHUNK_SIZE):
            yield self.dataset.select(range(i, min(i + CHUNK_SIZE, size)))

    def load_in_parallel(self, workers):
        """
        Farm chunk-processing out to a process pool. Big speed-up over serial,
        but it ties up the machine while running.
        """
        results = []
        chunk_count = (len(self.dataset) // CHUNK_SIZE) + 1
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in tqdm(pool.map(self.from_chunk, self.chunk_generator()), total=chunk_count):
                results.extend(batch)
        return results

    def load(self, workers=WORKERS):
        """Download this category from HuggingFace and curate it into a list of Items."""
        start = datetime.now()
        print(f"Loading dataset {self.category}", flush=True)
        self.dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{self.category}",
            split="full",
            trust_remote_code=True,
        )
        results = self.load_in_parallel(workers)
        finish = datetime.now()
        print(
            f"Completed {self.category} with {len(results):,} datapoints in "
            f"{(finish - start).total_seconds() / 60:.1f} mins",
            flush=True,
        )
        return results
