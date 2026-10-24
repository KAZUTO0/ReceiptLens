"""No-model smoke test for dataset formatting and evaluation code."""
from PIL import Image
from .data import training_conversation
from .metrics import score_prediction


def run_smoke():
    sample = {"image": Image.new("RGB", (100, 160), "white"),
              "entities": {"company": "Example", "date": "01/01/2019", "address": "Main Road", "total": "10.00"}}
    example = training_conversation(sample)
    assert example["messages"][0]["content"][1]["type"] == "image"
    raw = '{"company":"Example","date":"2019-01-01","address":"Main Road","total":"10"}'
    assert score_prediction(raw, sample["entities"])["all_four"] == 1
    print("ReceiptLens local smoke test passed (NO GPU/model used)")


if __name__ == "__main__":
    run_smoke()
