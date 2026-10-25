"""Safely parse the model output; malformed JSON does not turn into fake data."""

import json
from typing import Any
from .prompts import FIELDS


def parse_json_object(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            cleaned = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, TypeError):
        pass
    # Handles a preamble or trailing text without using fragile regex.
    decoder = json.JSONDecoder()
    for i, character in enumerate(cleaned):
        if character == "{":
            try:
                parsed, _ = decoder.raw_decode(cleaned[i:])
                if isinstance(parsed, dict):
                    return parsed
            except ValueError:
                continue
    return None


def parse_receipt(text: str) -> dict | None:
    """Returns canonical 4-field dictionary or None if not valid schema."""
    parsed = parse_json_object(text)
    if not isinstance(parsed, dict):
        return None
    if set(parsed) != set(FIELDS) or any(not isinstance(parsed[k], str) for k in FIELDS):
        return None
    return {key: parsed[key] for key in FIELDS}
