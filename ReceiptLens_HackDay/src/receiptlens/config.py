"""Single source of truth for the experiment configuration."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_id: str = "jsdnrs/ICDAR2019-SROIE"
    model_id: str = "unsloth/gemma-4-E2B-it"
    seed: int = 42
    train_count: int = 160
    validation_count: int = 20
    test_count: int = 30
    max_image_edge: int = 960
    max_new_tokens: int = 256
    lora_rank: int = 8
    lora_alpha: int = 8
    train_steps: int = 40
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation: int = 4
    output_dir: str = "outputs"

    def validate(self) -> None:
        if min(self.train_count, self.validation_count, self.test_count) < 1:
            raise ValueError("All sample counts must be positive")
        if self.max_image_edge < 256:
            raise ValueError("max_image_edge must be >= 256")
        if self.lora_rank < 1 or self.train_steps < 1:
            raise ValueError("LoRA rank and train steps must be positive")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
