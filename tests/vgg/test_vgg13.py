from unittest import TestCase

from convolut_zoo.cv.vgg import VGG13


class TestVGG13(TestCase):
    def test_pretrained(self):
        model = VGG13(pretrained=True)
        self.assertTrue(True)
