from unittest import TestCase

from convolut_zoo.cv.vgg import VGG19Bn


class TestVGG19Bn(TestCase):
    def test_pretrained(self):
        model = VGG19Bn(pretrained=True)
        self.assertTrue(True)
