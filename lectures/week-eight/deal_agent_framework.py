import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import chromadb
import numpy as np
from sklearn.manifold import TSNE

from agents.deals import Opportunity
from agents.planning_agent import PlanningAgent


BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"

CATEGORIES = [
    "Appliances",
    "Automotive",
    "Cell_Phones_and_Accessories",
    "Electronics",
    "Musical_Instruments",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
]
COLORS = ["red", "blue", "brown", "orange", "gold", "green", "purple", "cyan"]
CATEGORY_COLORS = dict(zip(CATEGORIES, COLORS))


def init_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if any(getattr(handler, "week8_console", False) for handler in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.week8_console = True
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S %z",
        )
    )
    root.addHandler(handler)


class DealAgentFramework:
    WEEK8_DIR = Path(__file__).resolve().parent
    DB = WEEK8_DIR / "products_vectorstore"
    MEMORY_FILENAME = WEEK8_DIR / "memory.json"

    def __init__(self):
        init_logging()
        self.client = chromadb.PersistentClient(path=str(self.DB))
        self.collection = self._get_populated_collection()
        self.memory = self.read_memory()
        self.planner = None

    def _get_populated_collection(self):
        counts = {
            collection.name: collection.count()
            for collection in self.client.list_collections()
        }
        name = next(
            (candidate for candidate in ("products", "products_lite") if counts.get(candidate, 0) > 0),
            None,
        )
        if name is None:
            raise RuntimeError(
                f"No populated Chroma product collection found in {self.DB}. "
                "Run the Day 2 vector-store build cells first."
            )
        self.log(f"Using Chroma collection {name!r} with {counts[name]:,} products")
        return self.client.get_collection(name)

    def init_agents_as_needed(self) -> None:
        if self.planner is None:
            self.log("Initializing Agent Framework")
            self.planner = PlanningAgent(self.collection)
            self.log("Agent Framework is ready")

    def read_memory(self) -> List[Opportunity]:
        if not self.MEMORY_FILENAME.exists():
            return []
        try:
            data = json.loads(self.MEMORY_FILENAME.read_text())
            return [Opportunity(**item) for item in data]
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self.log(f"Ignoring invalid memory file: {exc}")
            return []

    def write_memory(self) -> None:
        data = [opportunity.model_dump() for opportunity in self.memory]
        self.MEMORY_FILENAME.write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def reset_memory(cls, keep: int = 0) -> None:
        data = []
        if cls.MEMORY_FILENAME.exists():
            try:
                data = json.loads(cls.MEMORY_FILENAME.read_text())
            except json.JSONDecodeError:
                data = []
        cls.MEMORY_FILENAME.write_text(json.dumps(data[:keep], indent=2) + "\n")

    def log(self, message: str) -> None:
        logging.info(BG_BLUE + WHITE + "[Agent Framework] " + message + RESET)

    def run(self) -> List[Opportunity]:
        self.init_agents_as_needed()
        self.log("Kicking off Planning Agent")
        result = self.planner.plan(memory=self.memory)
        self.log(f"Planning Agent returned: {result}")
        if result and all(item.deal.url != result.deal.url for item in self.memory):
            self.memory.append(result)
            self.write_memory()
        return self.memory

    def get_plot_data(self, max_datapoints: int = 800) -> Tuple[List[str], np.ndarray, List[str]]:
        result = self.collection.get(
            include=["embeddings", "documents", "metadatas"],
            limit=max_datapoints,
        )
        vectors = np.asarray(result["embeddings"], dtype=float)
        if len(vectors) < 2:
            raise RuntimeError("At least two product embeddings are needed for the plot")
        perplexity = min(30, max(1, len(vectors) - 1))
        reduced = TSNE(
            n_components=3,
            random_state=42,
            perplexity=perplexity,
            init="random",
            learning_rate="auto",
        ).fit_transform(vectors)
        documents = result["documents"]
        colors = [
            CATEGORY_COLORS.get(metadata.get("category"), "gray")
            for metadata in result["metadatas"]
        ]
        return documents, reduced, colors


if __name__ == "__main__":
    DealAgentFramework().run()
