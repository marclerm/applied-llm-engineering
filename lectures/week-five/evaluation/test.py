"""
test.py — load the RAG evaluation test set.

Part of Ed Donner's evaluation package (week5/evaluation). Each line of
tests.jsonl is one TestQuestion. The only change from the course version is the
path: tests.jsonl lives in lectures/week-five/ (one level up from this package).

Adapted from ed-donner/llm_engineering (week5/evaluation/test.py).
"""

import json
from pathlib import Path
from pydantic import BaseModel, Field

# tests.jsonl sits in lectures/week-five/ (the parent of this evaluation/ package).
TEST_FILE = str(Path(__file__).parent.parent / "tests.jsonl")


class TestQuestion(BaseModel):
    """A test question with expected keywords and reference answer."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category (e.g., direct_fact, spanning, temporal)")


def load_tests() -> list[TestQuestion]:
    """Load test questions from JSONL file."""
    tests = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            tests.append(TestQuestion(**data))
    return tests
