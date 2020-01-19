import torch
import torch.nn as nn
from torch.nn import functional as F

from ..base import DecoderBlock, Conv3x3BnRelu, MaxPool2x2
from ..vgg import VGG11Bn


class UNet11Bn(nn.Module):
    def __init__(self,
                 num_classes=1,
                 num_filters=32,
                 pretrained=False,
                 is_deconv=True):
        super().__init__()

        encoder: VGG11Bn = VGG11Bn(pretrained=pretrained)

        self.conv1 = encoder.conv1
        self.conv2 = encoder.conv2
        self.conv3 = encoder.conv3
        self.conv4 = encoder.conv4
        self.conv5 = encoder.conv5

        self.pool = MaxPool2x2()

        self.center = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv=is_deconv)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv=is_deconv)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 4, is_deconv=is_deconv)
        self.dec3 = DecoderBlock(256 + num_filters * 4, num_filters * 4 * 2, num_filters * 2, is_deconv=is_deconv)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv=is_deconv)
        self.dec1 = Conv3x3BnRelu(64 + num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        self.final_sigm = nn.Sequential(nn.Conv2d(num_filters, num_classes, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x = F.log_softmax(self.final(dec1), dim=1)
        else:
            x = self.final_sigm(dec1)

        return x
