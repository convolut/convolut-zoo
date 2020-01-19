from unittest import TestCase

from convolut_zoo.cv.unet import UNet11Bn


class TestUNet11Bn(TestCase):
    def test_pretrained(self):
        model = UNet11Bn(pretrained=True)
        self.assertTrue(True)
