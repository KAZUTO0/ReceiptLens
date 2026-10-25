"""ReceiptLens: reproducible image-to-JSON receipt extraction experiment."""

from .config import ExperimentConfig
from .prompts import FIELDS, RECEIPT_INSTRUCTION

__all__ = ["ExperimentConfig", "FIELDS", "RECEIPT_INSTRUCTION"]
