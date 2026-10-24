"""Deterministic normalized exact-match scores for the receipt dataset."""

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
import pandas as pd
from .extract import parse_json_object, parse_receipt
from .prompts import FIELDS


def normalize_text(value: str) -> str:
    """Case-folding and normalized whitespace only; punctuation remains meaningful."""
    return " ".join(str(value).casefold().split())


def normalize_date(value: str) -> str:
    text = str(value).strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return normalize_text(text)


def normalize_money(value: str) -> str:
    value = re.sub(r"(?i)\b(?:rm|myr|usd|inr)\b", "", str(value))
    value = re.sub(r"[₹$\s,]", "", value).strip()
    try:
        return str(Decimal(value).quantize(Decimal("0.01")))
    except (InvalidOperation, ValueError):
        return normalize_text(value)


def field_equal(predicted, actual, field: str) -> bool:
    normalizer = normalize_date if field == "date" else normalize_money if field == "total" else normalize_text
    return normalizer(predicted) == normalizer(actual)


def score_prediction(raw_text: str, ground_truth: dict) -> dict:
    """Missing/malformed JSON receives zero field credits, even for blank truths."""
    valid_json = parse_json_object(raw_text) is not None
    parsed = parse_receipt(raw_text)
    values = {
        k: int(field_equal(parsed[k], ground_truth[k], k)) if parsed is not None else 0
        for k in FIELDS
    }
    return {
        "valid_json": int(valid_json),
        "valid_schema": int(parsed is not None),
        **values,
        "all_four": int(all(values.values())),
    }


def evaluate_samples(dataset, predict_fn, *, label: str, max_count: int | None = None):
    """Run images through prediction callback; no fake predictions or leaked entities."""
    rows = []
    n = min(len(dataset), max_count) if max_count is not None else len(dataset)
    for idx in range(n):
        sample = dataset[idx]
        gold = {field: str(sample["entities"].get(field) or "") for field in FIELDS}
        try:
            raw = str(predict_fn(sample["image"]))
            err = ""
        except Exception as exc:
            raw = ""
            err = f"{type(exc).__name__}: {exc}"
        metrics = score_prediction(raw, gold)
        rows.append({"index": idx, "model": label, "prediction": raw,
                     "ground_truth": __import__("json").dumps(gold, ensure_ascii=False),
                     "error": err, **metrics})
        if (idx + 1) % 5 == 0 or idx + 1 == n:
            print(f"{label}: {idx + 1}/{n} evaluated")
    return pd.DataFrame(rows)


def compare_results(base: pd.DataFrame, tuned: pd.DataFrame) -> pd.DataFrame:
    if list(base["index"]) != list(tuned["index"]):
        raise ValueError("Baseline and fine-tuned result rows do not align")
    cols = ["valid_json", "valid_schema", *FIELDS, "all_four"]
    result = pd.DataFrame({
        "Plain (%)": base[cols].mean().mul(100),
        "Fine-tuned (%)": tuned[cols].mean().mul(100),
    }).round(1)
    result["Change (pp)"] = (result["Fine-tuned (%)"] - result["Plain (%)"]).round(1)
    result.index.name = "metric"
    return result
