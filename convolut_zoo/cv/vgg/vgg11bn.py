from torch import nn

from ..base import MaxPool2x2, Conv3x3BnRelu
from .base import VGG


class VGG11Bn(VGG):
    def __init__(self,
                 num_classes=1000,
                 pretrained=False):
        conv1 = nn.Sequential(Conv3x3BnRelu(3, 64))
        conv2 = nn.Sequential(Conv3x3BnRelu(64, 128))
        conv3 = nn.Sequential(Conv3x3BnRelu(128, 256),
                              Conv3x3BnRelu(256, 256))
        conv4 = nn.Sequential(Conv3x3BnRelu(256, 512),
                              Conv3x3BnRelu(512, 512))
        conv5 = nn.Sequential(Conv3x3BnRelu(512, 512),
                              Conv3x3BnRelu(512, 512))

        features = nn.Sequential(
            conv1,
            MaxPool2x2(),
            conv2,
            MaxPool2x2(),
            conv3,
            MaxPool2x2(),
            conv4,
            MaxPool2x2(),
            conv5,
            MaxPool2x2(),
        )
        mapping = {
            'features.0': 'features.0.0.conv',
            'features.1': 'features.0.0.batch_norm',

            'features.4': 'features.2.0.conv',
            'features.5': 'features.2.0.batch_norm',

            'features.8': 'features.4.0.conv',
            'features.9': 'features.4.0.batch_norm',
            'features.11': 'features.4.1.conv',
            'features.12': 'features.4.1.batch_norm',

            'features.15': 'features.6.0.conv',
            'features.16': 'features.6.0.batch_norm',
            'features.18': 'features.6.1.conv',
            'features.19': 'features.6.1.batch_norm',

            'features.22': 'features.8.0.conv',
            'features.23': 'features.8.0.batch_norm',
            'features.25': 'features.8.1.conv',
            'features.26': 'features.8.1.batch_norm',
        }

        super().__init__(features, num_classes, not pretrained, pretrained, mapping)
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.conv4 = conv4
        self.conv5 = conv5
