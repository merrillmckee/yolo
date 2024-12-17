import torch.nn as nn

from models.cba import CBA


class BottleneckYoloV3(nn.Module):
    """
    Create a YoloV3 style bottleneck  # https://arxiv.org/abs/1804.02767
    """
    def __init__(
            self,
            in_out_channels: int,
            ratio: float = 1.0,
    ):
        super().__init__()
        hidden_channels = int(in_out_channels * ratio)
        self.cba1 = CBA(in_out_channels, hidden_channels, kernel_size=1)
        self.cba2 = CBA(hidden_channels, in_out_channels, kernel_size=3)

    def forward(self, x):
        return x + self.cba2(self.cba1(x))


class BottleneckResnet(nn.Module):
    """
    Create a Resnet style bottleneck  # https://arxiv.org/abs/1804.02767
    """
    def __init__(
            self,
            in_out_channels: int,
            ratio: float = 1.0,
    ):
        super().__init__()
        hidden_channels = int(in_out_channels * ratio)
        self.cba1 = CBA(in_out_channels, hidden_channels, kernel_size=1)
        self.cba2 = CBA(hidden_channels, hidden_channels, kernel_size=3)
        self.cba3 = CBA(hidden_channels, in_out_channels, kernel_size=1)

    def forward(self, x):
        return x + self.cba3(self.cba2(self.cba1(x)))


class BottleneckGroup(nn.Module):
    """
    Create group of successive bottlenecks; mentioned in YoloV3  # https://arxiv.org/abs/1804.02767
    """
    def __init__(
            self,
            in_out_channels: int,
            n_groups: int,
    ):
        super().__init__()
        self.cba1 = CBA(in_out_channels, in_out_channels, kernel_size=1)
        self.group = nn.Sequential(*(BottleneckYoloV3(in_out_channels=in_out_channels) for _ in range(n_groups)))

    def forward(self, x):
        return self.cba1(self.group(x))
