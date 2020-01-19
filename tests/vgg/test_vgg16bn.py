from unittest import TestCase

from convolut_zoo.cv.vgg import VGG16Bn


class TestVGG16Bn(TestCase):
    def test_pretrained(self):
        model = VGG16Bn(pretrained=True)
        self.assertTrue(True)
