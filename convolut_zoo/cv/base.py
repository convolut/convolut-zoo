import torch

from torch import nn


class Conv3x3(nn.Conv2d):
    def __init__(self, in_: int, out: int):
        super().__init__(in_, out, kernel_size=3, padding=1)

    def initialize(self, mode: str = 'fan_out', nonlinearity: str = 'relu', bias: int = 0):
        nn.init.kaiming_normal(self.weight, mode=mode, nonlinearity=nonlinearity)
        if self.bias is not None:
            nn.init.constant_(self.bias, bias)


class Conv3x3Relu(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.conv = Conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)

        return x

    def initialize(self, mode: str = 'fan_out', nonlinearity: str = 'relu', bias: int = 0):
        self.conv.initialize()


class Conv3x3BnRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.conv = Conv3x3(in_, out)
        self.batch_norm = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        return x

    def initialize(self, mode: str = 'fan_out', nonlinearity: str = 'relu', bias: int = 0):
        self.conv.initialize()


class MaxPool2x2(nn.MaxPool2d):
    def __init__(self):
        super().__init__(kernel_size=2, stride=2)


class DecoderBlock(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                Conv3x3BnRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                Conv3x3BnRelu(in_channels, middle_channels),
                Conv3x3BnRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)
