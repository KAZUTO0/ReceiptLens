from PIL import Image
from receiptlens.demo import image_digest

def test_hash_is_stable_for_same_image():
    im=Image.new('RGB',(100,100),'white')
    assert image_digest(im)==image_digest(im.copy())
