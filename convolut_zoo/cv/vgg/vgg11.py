from torch import nn

from ..base import Conv3x3Relu, MaxPool2x2
from .base import VGG


class VGG11(VGG):
    def __init__(self,
                 num_classes=1000,
                 pretrained=False):
        conv1 = nn.Sequential(Conv3x3Relu(3, 64))
        conv2 = nn.Sequential(Conv3x3Relu(64, 128))
        conv3 = nn.Sequential(Conv3x3Relu(128, 256),
                              Conv3x3Relu(256, 256))
        conv4 = nn.Sequential(Conv3x3Relu(256, 512),
                              Conv3x3Relu(512, 512))
        conv5 = nn.Sequential(Conv3x3Relu(512, 512),
                              Conv3x3Relu(512, 512))

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

            'features.3': 'features.2.0.conv',

            'features.6': 'features.4.0.conv',
            'features.8': 'features.4.1.conv',

            'features.11': 'features.6.0.conv',
            'features.13': 'features.6.1.conv',

            'features.16': 'features.8.0.conv',
            'features.18': 'features.8.1.conv',
        }

        super().__init__(features, num_classes, not pretrained, pretrained, mapping)
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.conv4 = conv4
        self.conv5 = conv5
