"""
parser.py — turn a raw Amazon datapoint into a clean Item (or drop it).

This is the heart of data curation. For each raw product we:
  - keep only items in a sensible price band ($0.50–$999.49),
  - scrub the text (strip product/model numbers and noisy fields, cap length),
  - require enough text to be useful (MIN_CHARS),
  - extract a weight in pounds where possible.

Datapoints that don't qualify return None so the caller can filter them out.

Adapted from ed-donner/llm_engineering (week6/pricer/parser.py).
"""

from pricer.items import Item
import json
import re

# Curation thresholds.
MIN_CHARS = 600        # require at least this much text, or the item isn't informative enough
MIN_PRICE = 0.5        # ignore freebies / mispriced items
MAX_PRICE = 999.49     # cap so the model learns a bounded $0–$999 range
MAX_TEXT_EACH = 3000   # max chars taken from any single field (description/features/details)
MAX_TEXT_TOTAL = 4000  # max chars for the whole scrubbed string

# Boilerplate detail keys that add noise, not signal — removed before building text.
REMOVALS = [
    "Part Number",
    "Best Sellers Rank",
    "Batteries Included?",
    "Batteries Required?",
    "Item model number",
]


def simplify(text_list) -> str:
    """Flatten a value to a single tidy line, collapsing whitespace and capping length."""
    return (
        str(text_list)
        .replace("\n", " ")
        .replace("\r", "")
        .replace("\t", "")
        .replace("  ", " ")
        .strip()[:MAX_TEXT_EACH]
    )


def scrub(title, description, features, details) -> str:
    """
    Build a cleansed product description.

    Joins title + description + features + remaining details, then strips out
    long alphanumeric product/model codes (e.g. "B01D05U9NO") with a regex —
    those are pure noise for a price model — and caps the total length.
    """
    for remove in REMOVALS:
        details.pop(remove, None)
    result = title + "\n"
    if description:
        result += simplify(description) + "\n"
    if features:
        result += simplify(features) + "\n"
    if details:
        result += json.dumps(details) + "\n"
    # Match 7+ char tokens that contain at least one letter AND one digit (product codes).
    pattern = r"\b(?=[A-Z0-9]{7,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b"
    return re.sub(pattern, "", result).strip()[:MAX_TEXT_TOTAL]


def get_weight(details):
    """Extract the item weight in POUNDS from the details dict, converting units as needed."""
    weight_str = details.get("Item Weight")
    if weight_str:
        parts = weight_str.split(" ")
        amount = float(parts[0])
        unit = parts[1].lower()
        if unit == "pounds":
            return amount
        elif unit == "ounces":
            return amount / 16
        elif unit == "grams":
            return amount / 453.592
        elif unit == "milligrams":
            return amount / 453592
        elif unit == "kilograms":
            return amount / 0.453592
        elif unit == "hundredths" and parts[2].lower() == "pounds":
            return amount / 100
    return 0


def parse(datapoint, category):
    """
    Convert one raw datapoint into an Item, or return None if it fails curation.

    Filters on price band and minimum text length; attaches the scrubbed text,
    category and weight.
    """
    try:
        price = float(datapoint["price"])
    except ValueError:
        return None
    if MIN_PRICE <= price <= MAX_PRICE:
        title = datapoint["title"]
        description = datapoint["description"]
        features = datapoint["features"]
        details = json.loads(datapoint["details"])
        weight = get_weight(details)
        full = scrub(title, description, features, details)
        if len(full) >= MIN_CHARS:
            return Item(
                title=title,
                category=category,
                price=price,
                full=full,
                weight=weight,
            )
