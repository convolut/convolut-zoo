from unittest import TestCase

from convolut_zoo.cv.vgg import VGG16


class TestVGG16(TestCase):
    def test_pretrained(self):
        model = VGG16(pretrained=True)
        self.assertTrue(True)
