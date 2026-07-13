"""
items.py — the Item data model for the "Price is Right" project.

Week 7 extension: adds tokenizer-aware prompt generation for open-source
model fine-tuning (QLoRA), and HuggingFace Hub helpers to push/load
the prompt-formatted dataset used by the SFT trainer.

Adapted from ed-donner/llm_engineering (week7/pricer/items.py).
"""

from pydantic import BaseModel
from datasets import Dataset, DatasetDict, load_dataset
from typing import Optional, Self


PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"


class Item(BaseModel):
    """
    An Item is a data-point of a Product with a Price.
    """

    title: str
    category: str
    price: float
    full: Optional[str] = None
    weight: Optional[float] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    completion: Optional[str] = None
    id: Optional[int] = None

    def make_prompt(self, text: str):
        """Build a training prompt ending with the rounded price."""
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}{round(self.price)}.00"

    def test_prompt(self) -> str:
        """The prompt with the price stripped — what we feed the model at inference time."""
        return self.prompt.split(PREFIX)[0] + PREFIX

    def __repr__(self) -> str:
        return f"<{self.title} = ${self.price}>"

    def count_tokens(self, tokenizer) -> int:
        """Count tokens in the summary (used to pick a safe truncation cutoff)."""
        return len(tokenizer.encode(self.summary, add_special_tokens=False))

    def make_prompts(self, tokenizer, max_tokens: int, include_price: bool):
        """Build self.prompt + self.completion, truncating the summary to max_tokens.

        For train/val (include_price=True) the completion is the rounded dollar amount
        the model must learn. For test (include_price=False) the completion is the raw
        float price, kept for evaluation only.
        """
        tokens = tokenizer.encode(self.summary, add_special_tokens=False)
        summary = (
            tokenizer.decode(tokens[:max_tokens]).rstrip()
            if len(tokens) > max_tokens
            else self.summary
        )
        self.prompt = f"{QUESTION}\n\n{summary}\n\n{PREFIX}"
        self.completion = f"{round(self.price)}.00" if include_price else str(self.price)

    def count_prompt_tokens(self, tokenizer) -> int:
        """Count tokens in the combined prompt + completion."""
        return len(tokenizer.encode(self.prompt + self.completion, add_special_tokens=False))

    def to_datapoint(self) -> dict:
        return {"prompt": self.prompt, "completion": self.completion}

    @staticmethod
    def push_to_hub(dataset_name: str, train: list[Self], val: list[Self], test: list[Self]):
        """Push Item lists to the HuggingFace Hub as a DatasetDict."""
        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]),
                "validation": Dataset.from_list([item.model_dump() for item in val]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        """Load a DatasetDict from the HuggingFace Hub and reconstruct (train, val, test) Items."""
        ds = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )

    @staticmethod
    def push_prompts_to_hub(
        dataset_name: str, train: list[Self], val: list[Self], test: list[Self]
    ):
        """Push prompt/completion pairs to the HuggingFace Hub for SFT training."""
        DatasetDict(
            {
                "train": Dataset.from_list([item.to_datapoint() for item in train]),
                "val": Dataset.from_list([item.to_datapoint() for item in val]),
                "test": Dataset.from_list([item.to_datapoint() for item in test]),
            }
        ).push_to_hub(dataset_name)
