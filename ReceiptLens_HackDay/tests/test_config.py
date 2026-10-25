import json
import pytest
from receiptlens.config import ExperimentConfig

def test_config_serialization(tmp_path):
    c=ExperimentConfig()
    c.validate()
    c.save(tmp_path/'config.json')
    assert json.loads((tmp_path/'config.json').read_text())['model_id']==c.model_id

def test_config_invalid():
    with pytest.raises(ValueError):
        ExperimentConfig(train_count=0).validate()
