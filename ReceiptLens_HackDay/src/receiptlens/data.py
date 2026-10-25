"""SROIE data ingestion. Ground-truth entities are NEVER used as model inputs."""

from typing import Mapping
import json
from PIL import Image, ImageOps
from .prompts import FIELDS, RECEIPT_INSTRUCTION


def load_sroie(dataset_id: str):
    from datasets import load_dataset
    raw = load_dataset(dataset_id)
    if "train" not in raw or "test" not in raw:
        raise ValueError(f"Expected train/test splits, got {list(raw.keys())}")
    for split_name in ("train", "test"):
        missing = {"image", "entities"} - set(raw[split_name].column_names)
        if missing:
            raise ValueError(f"Missing dataset columns in {split_name}: {sorted(missing)}")
    return raw


def select_splits(raw, train_count=160, validation_count=20, test_count=30, seed=42):
    """Split *only* the official training set into train and validation.

    The official test split is reserved for final evaluation and is never
    used for training or hyperparameter selection.
    """
    if train_count + validation_count > len(raw["train"]):
        raise ValueError("Requested more train+validation items than available")
    if test_count > len(raw["test"]):
        raise ValueError("Requested more test items than available")
    shuffled_train = raw["train"].shuffle(seed=seed)
    shuffled_test = raw["test"].shuffle(seed=seed + 1)
    return {
        "train": shuffled_train.select(range(train_count)),
        "validation": shuffled_train.select(range(train_count, train_count + validation_count)),
        "test": shuffled_test.select(range(test_count)),
    }


def prepare_image(image: Image.Image, max_edge: int = 960) -> Image.Image:
    """Resize without distorting the receipt; avoid excessive downscaling of text."""
    if image is None:
        raise ValueError("No image supplied")
    img = ImageOps.exif_transpose(image).convert("RGB")
    if max(img.size) > max_edge:
        img.thumbnail((max_edge, max_edge), resample=Image.Resampling.LANCZOS)
    return img


def gold_entities(sample: Mapping) -> dict[str, str]:
    entities = sample["entities"]
    # Strictly separate annotation -> target labels from image -> model input.
    return {name: str(entities.get(name) or "") for name in FIELDS}


def training_conversation(sample: Mapping, max_edge: int = 960) -> dict:
    """Unsloth vision supervised fine-tuning format."""
    return {"messages": [
        {"role": "user", "content": [
            {"type": "text", "text": RECEIPT_INSTRUCTION},
            {"type": "image", "image": prepare_image(sample["image"], max_edge)},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": json.dumps(gold_entities(sample), ensure_ascii=False)}
        ]},
    ]}


def training_examples(dataset, max_edge: int = 960):
    return [training_conversation(example, max_edge) for example in dataset]
