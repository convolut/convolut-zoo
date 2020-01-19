from unittest import TestCase

from convolut_zoo.cv.vgg import VGG13Bn


class TestVGG13(TestCase):
    def test_pretrained(self):
        model = VGG13Bn(pretrained=True)
        self.assertTrue(True)
