from PIL import Image
from receiptlens.data import prepare_image, training_conversation, gold_entities


def test_resize_keeps_aspect_ratio():
    image = prepare_image(Image.new('RGBA',(400,1600)), max_edge=800)
    assert image.size == (200,800)
    assert image.mode == 'RGB'


def test_annotations_only_in_target():
    sample = {'image':Image.new('RGB',(200,300)),
              'entities':{'company':'SECRET MERCHANT', 'date':'01/01/2019','address':'ROAD','total':'5'}}
    messages = training_conversation(sample)['messages']
    assert 'SECRET MERCHANT' not in messages[0]['content'][0]['text']
    assert 'SECRET MERCHANT' in messages[1]['content'][0]['text']
