from torch import nn

from ..base import Conv3x3BnRelu, MaxPool2x2
from .base import VGG


class VGG19Bn(VGG):
    def __init__(self,
                 num_classes=1000,
                 pretrained=False):
        conv1 = nn.Sequential(Conv3x3BnRelu(3, 64),
                              Conv3x3BnRelu(64, 64))
        conv2 = nn.Sequential(Conv3x3BnRelu(64, 128),
                              Conv3x3BnRelu(128, 128))
        conv3 = nn.Sequential(Conv3x3BnRelu(128, 256),
                              Conv3x3BnRelu(256, 256),
                              Conv3x3BnRelu(256, 256),
                              Conv3x3BnRelu(256, 256))
        conv4 = nn.Sequential(Conv3x3BnRelu(256, 512),
                              Conv3x3BnRelu(512, 512),
                              Conv3x3BnRelu(512, 512),
                              Conv3x3BnRelu(512, 512))
        conv5 = nn.Sequential(Conv3x3BnRelu(512, 512),
                              Conv3x3BnRelu(512, 512),
                              Conv3x3BnRelu(512, 512),
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
            'features.3': 'features.0.1.conv',
            'features.4': 'features.0.1.batch_norm',

            'features.7': 'features.2.0.conv',
            'features.8': 'features.2.0.batch_norm',
            'features.10': 'features.2.1.conv',
            'features.11': 'features.2.1.batch_norm',

            'features.14': 'features.4.0.conv',
            'features.15': 'features.4.0.batch_norm',
            'features.17': 'features.4.1.conv',
            'features.18': 'features.4.1.batch_norm',
            'features.20': 'features.4.2.conv',
            'features.21': 'features.4.2.batch_norm',
            'features.23': 'features.4.3.conv',
            'features.24': 'features.4.3.batch_norm',

            'features.27': 'features.6.0.conv',
            'features.28': 'features.6.0.batch_norm',
            'features.30': 'features.6.1.conv',
            'features.31': 'features.6.1.batch_norm',
            'features.33': 'features.6.2.conv',
            'features.34': 'features.6.2.batch_norm',
            'features.36': 'features.6.3.conv',
            'features.37': 'features.6.3.batch_norm',

            'features.40': 'features.8.0.conv',
            'features.41': 'features.8.0.batch_norm',
            'features.43': 'features.8.1.conv',
            'features.44': 'features.8.1.batch_norm',
            'features.46': 'features.8.2.conv',
            'features.47': 'features.8.2.batch_norm',
            'features.49': 'features.8.3.conv',
            'features.50': 'features.8.3.batch_norm',
        }

        super().__init__(features, num_classes, not pretrained, pretrained, mapping)
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.conv4 = conv4
        self.conv5 = conv5
