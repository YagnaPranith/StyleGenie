"""
recommend.py
------------
Loads fashion_dataset.csv and returns the best matching outfit row for a
given (age_group, gender, occasion) combination.

Matching strategy (in priority order):
  1. Exact age_group + gender + occasion match  → pick first row
  2. Adjacent age group + same gender + occasion → fallback
  3. Same gender + occasion (any age)            → last resort fallback
  4. Hard-coded generic fallback                 → should never be reached

The CSV has 9 occasions × 2 genders × 12 age groups = 216 rows minimum,
so an exact match will almost always be found.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Canonical occasion list (shown in the UI dropdown) ───────────────────────
OCCASIONS = [
    "Birthday",
    "Beach",
    "Wedding",
    "Get Together",
    "Function",
    "Temple",
    "Office",
    "Picnic",
    "Park",
]

# ── Age bucket ordering (for adjacent-group fallback) ────────────────────────
AGE_ORDER = [
    "0-4", "5-9", "10-14", "15-19", "20-24", "25-29",
    "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
]

DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "fashion_dataset.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loader (cached at module level — loaded once on first import)
# ─────────────────────────────────────────────────────────────────────────────

_ROWS: list[dict] = []


def _load_dataset() -> list[dict]:
    global _ROWS
    if _ROWS:
        return _ROWS
    if not DATASET_PATH.exists():
        logger.error("fashion_dataset.csv not found at %s", DATASET_PATH)
        return []
    with DATASET_PATH.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _ROWS = [row for row in reader]
    logger.info("Loaded %d outfit rows from dataset", len(_ROWS))
    return _ROWS


# ─────────────────────────────────────────────────────────────────────────────
# Matching helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return text.strip().lower()


def _find_row(
    rows: list[dict],
    age_group: str,
    gender: str,
    occasion: str,
) -> dict | None:
    ag = _normalize(age_group)
    ge = _normalize(gender)
    oc = _normalize(occasion)

    for row in rows:
        if (
            _normalize(row.get("Age_Group", "")) == ag
            and _normalize(row.get("Gender", "")) == ge
            and _normalize(row.get("Occasion", "")) == oc
        ):
            return row
    return None


def _adjacent_age_groups(age_group: str) -> list[str]:
    """Return age groups adjacent to the given one, nearest first."""
    try:
        idx = AGE_ORDER.index(age_group)
    except ValueError:
        return AGE_ORDER  # unknown group → try all

    result = []
    lo, hi = idx - 1, idx + 1
    while lo >= 0 or hi < len(AGE_ORDER):
        if hi < len(AGE_ORDER):
            result.append(AGE_ORDER[hi])
            hi += 1
        if lo >= 0:
            result.append(AGE_ORDER[lo])
            lo -= 1
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def recommend_outfit(
    age_group: str,
    gender: str,
    skin_tone: str,
    occasion: str,
) -> dict:
    """
    Return a recommendation dict for the given attributes.

    The returned dict has these keys (all str):
        occasion, age_group, gender, skin_tone,
        top_wear, bottom_wear, footwear, accessories,
        color_combination, style_tip
    """
    rows = _load_dataset()

    # ── 1. Exact match ─────────────────────────────────────────────────────
    row = _find_row(rows, age_group, gender, occasion)

    # ── 2. Adjacent age group ───────────────────────────────────────────────
    if row is None:
        for adjacent in _adjacent_age_groups(age_group):
            row = _find_row(rows, adjacent, gender, occasion)
            if row:
                logger.info(
                    "Using adjacent age group '%s' instead of '%s'",
                    adjacent, age_group,
                )
                break

    # ── 3. Any age, same gender + occasion ─────────────────────────────────
    if row is None:
        ge = _normalize(gender)
        oc = _normalize(occasion)
        for r in rows:
            if (
                _normalize(r.get("Gender", "")) == ge
                and _normalize(r.get("Occasion", "")) == oc
            ):
                row = r
                logger.info("Fell back to any-age match for %s / %s", gender, occasion)
                break

    # ── 4. Hard-coded last resort ───────────────────────────────────────────
    if row is None:
        logger.warning("No dataset match for %s / %s / %s", age_group, gender, occasion)
        row = _generic_fallback(gender, occasion)

    # ── Build response dict ─────────────────────────────────────────────────
    return {
        "occasion":         occasion,
        "age_group":        row.get("Age_Group", age_group),
        "gender":           row.get("Gender", gender),
        "skin_tone":        skin_tone,
        "top_wear":         row.get("Top_Wear", "—"),
        "bottom_wear":      row.get("Bottom_Wear", "—"),
        "footwear":         row.get("Footwear", "—"),
        "accessories":      row.get("Accessories", "—"),
        "color_combination": row.get("Color_Combination", "—"),
        "style_tip":        row.get("Style_Tip", "Dress with confidence."),
    }


def _generic_fallback(gender: str, occasion: str) -> dict:
    """Absolute last-resort response when the CSV has no usable row."""
    if gender.lower() == "female":
        return {
            "Age_Group": "—",
            "Gender": gender,
            "Occasion": occasion,
            "Top_Wear": "Elegant solid-colored blouse",
            "Bottom_Wear": "Tailored straight trousers",
            "Footwear": "Block-heeled sandals",
            "Accessories": "Simple gold earrings & watch",
            "Color_Combination": "Ivory & navy",
            "Style_Tip": "Keep it simple — a well-fitted outfit in a neutral palette always works.",
        }
    return {
        "Age_Group": "—",
        "Gender": gender,
        "Occasion": occasion,
        "Top_Wear": "Classic Oxford shirt",
        "Bottom_Wear": "Slim-fit formal trousers",
        "Footwear": "Polished Oxford shoes",
        "Accessories": "Leather belt & analog watch",
        "Color_Combination": "White & charcoal",
        "Style_Tip": "A well-pressed shirt and slim-fit trousers is a universally reliable choice.",
    }
