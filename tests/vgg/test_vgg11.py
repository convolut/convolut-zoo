from unittest import TestCase
from convolut_zoo.cv.vgg import VGG11


class TestVGG11(TestCase):
    def test_pretrained(self):
        model: VGG11 = VGG11(pretrained=True)
        self.assertTrue(True)
