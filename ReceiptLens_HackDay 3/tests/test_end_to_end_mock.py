"""End-to-end offline dry run: one image -> prediction callback -> scored CSV.

No model weights, no GPU, and no fabricated competition performance here.
"""
from PIL import Image
from receiptlens.metrics import evaluate_samples
from receiptlens.config import ExperimentConfig
from receiptlens.reports import save_report


class MockDataset:
    def __len__(self):
        return 2

    def __getitem__(self, i):
        return {"image": Image.new("RGB", (120, 180)),
                "entities": {"company": "A" if i == 0 else "B",
                             "date": "10/10/2018", "address": "Road 1", "total": "10.50"}}


def test_fake_predictions_export(tmp_path):
    # A deliberate mock predictor verifies wiring, NOT model accuracy.
    mock = lambda image: '{"company":"A","date":"10/10/2018","address":"Road 1","total":"10.50"}'
    baseline = evaluate_samples(MockDataset(), mock, label="mock_plain")
    finetuned = evaluate_samples(MockDataset(), mock, label="mock_tuned")
    compare = save_report(baseline, finetuned, ExperimentConfig(), tmp_path)
    assert (tmp_path / "baseline_predictions.csv").exists()
    assert (tmp_path / "finetuned_predictions.csv").exists()
    assert (tmp_path / "comparison.csv").exists()
    assert compare.loc["company", "Plain (%)"] == 50.0
    assert compare.loc["company", "Change (pp)"] == 0.0
