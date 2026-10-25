import json
import pandas as pd
import pytest
from receiptlens.metrics import score_prediction, normalize_money, compare_results
G = {'company':'SHOP','date':'05/11/2018','address':'Main Road', 'total':'1,234.50'}

def test_normalized_predictions():
    raw = json.dumps({'company':'shop','date':'2018-11-05','address':' Main   Road ','total':'RM 1234.5'})
    assert score_prediction(raw,G)['all_four'] == 1

def test_invalid_output_cannot_win_for_blank_truth():
    truth = {**G,'address':''}
    assert score_prediction('garbage',truth)['address'] == 0
    assert score_prediction('garbage',truth)['valid_schema'] == 0

def test_comparison_requires_same_rows():
    left = pd.DataFrame({'index':[0],'valid_json':[1], 'valid_schema':[1],
                         'company':[1], 'date':[1], 'address':[1], 'total':[1], 'all_four':[1]})
    right = left.copy()
    assert compare_results(left,right).loc['company','Plain (%)'] == 100
    with pytest.raises(ValueError):
        compare_results(left, right.assign(index=[2]))
