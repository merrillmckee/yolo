import torch.nn as nn


class CBA(nn.Module):
    """
    Convolution, batch normalization, activation building block
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm2d(out_channels)  # eps, momentum
        self.act = nn.LeakyReLU()  # negative_slope

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
