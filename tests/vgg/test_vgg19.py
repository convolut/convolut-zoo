from unittest import TestCase

from convolut_zoo.cv.vgg import VGG19


class TestVGG19(TestCase):
    def test_pretrained(self):
        model = VGG19(pretrained=True)
        self.assertTrue(True)
