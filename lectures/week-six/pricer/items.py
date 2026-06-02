"""
items.py — the Item data model for the "Price is Right" project.

An Item is one curated data-point: a product (title, category, cleaned text,
weight) paired with its price. We also build the training prompt here — the
exact text the model will learn from / be tested on — and provide helpers to
push the curated dataset to (and load it back from) the HuggingFace Hub.

Adapted from ed-donner/llm_engineering (week6/pricer/items.py).
"""

from pydantic import BaseModel
from datasets import Dataset, DatasetDict, load_dataset
from typing import Optional, Self


# The training prompt is built as: QUESTION + the product text + PREFIX + price.
# At inference time we cut the prompt at PREFIX, so the model has to *complete*
# the price — which is how we both train and evaluate it.
PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"


class Item(BaseModel):
    """
    An Item is a data-point of a Product with a Price.
    """

    title: str
    category: str
    price: float
    full: Optional[str] = None      # the cleaned/scrubbed product text
    weight: Optional[float] = None  # in pounds (see parser.get_weight)
    summary: Optional[str] = None
    prompt: Optional[str] = None    # the full training prompt (question + text + price)
    id: Optional[int] = None

    def make_prompt(self, text: str):
        """Build the training prompt, ending with the rounded price the model must learn."""
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}{round(self.price)}.00"

    def test_prompt(self) -> str:
        """The prompt with the price removed — what we feed the model at test time."""
        return self.prompt.split(PREFIX)[0] + PREFIX

    def __repr__(self) -> str:
        return f"<{self.title} = ${self.price}>"

    @staticmethod
    def push_to_hub(dataset_name: str, train: list[Self], val: list[Self], test: list[Self]):
        """Push train/validation/test Item lists to the HuggingFace Hub as a DatasetDict."""
        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]),
                "validation": Dataset.from_list([item.model_dump() for item in val]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        """Load a dataset from the HuggingFace Hub and reconstruct (train, val, test) Items."""
        ds = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )
