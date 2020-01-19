from unittest import TestCase

from convolut_zoo.cv.vgg import VGG11Bn


class TestVGG11Bn(TestCase):
    def test_pretrained(self):
        model = VGG11Bn(pretrained=True)
        self.assertTrue(True)
