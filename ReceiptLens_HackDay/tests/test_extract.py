import json
from receiptlens.extract import parse_json_object, parse_receipt

VALID = {'company':'Store','date':'02/03/2019','address':'No 1 Main Road','total':'9.10'}

def test_code_fences_and_prose():
    assert parse_receipt('```json\n'+json.dumps(VALID)+'\n```') == VALID
    assert parse_receipt('Here:\n'+json.dumps(VALID)+'\nThank you') == VALID

def test_invalid_and_missing_schema():
    assert parse_receipt('bad output') is None
    assert parse_receipt('{"company":"Store"}') is None
    assert parse_receipt('[]') is None
    assert parse_receipt(json.dumps({**VALID, 'total':9.1})) is None

def test_empty_string_allowed_but_all_keys_required():
    assert parse_receipt(json.dumps({**VALID, 'address':''}))['address'] == ''
