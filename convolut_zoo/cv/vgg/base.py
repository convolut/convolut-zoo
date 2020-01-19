from typing import Dict, Any

import torch
from torch import nn
from torch.hub import load_state_dict_from_url

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):
    def __init__(self,
                 features,
                 num_classes=1000,
                 init_weights=True,
                 pretrained=False,
                 mapping: Dict[str, str] = None):
        super().__init__()
        self.mapping = mapping
        self.name = type(self).__name__

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[self.name.lower()])
            self.load_state_dict(self._map(state_dict))

    def _map(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self.mapping:
            mapping = {}

            for key in state_dict.keys():
                for old, new in self.mapping.items():
                    if old in key:
                        mapping[key] = key.replace(old, new)

            for old, new in mapping.items():
                state_dict[new] = state_dict.pop(old)

        return state_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

            initiailize = getattr(self, "initiailize", None)
            if callable(initiailize):
                initiailize()
