import torch.nn as nn

from models.bottleneck import BottleneckGroup
from models.cba import CBA


class Objects(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, num_anchors: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_outputs = 5 + num_classes  # 5: (x, y, w, h, p_obj)
        self.cba1 = CBA(in_channels, self.num_anchors * self.num_outputs, kernel_size=1)

    def forward(self, x):
        output = self.cba1(x)
        batch_size = output.shape[0]
        img_size = output.shape[2]
        output_reshaped = output.reshape((batch_size, self.num_anchors, img_size, img_size, self.num_outputs))
        return output_reshaped


class YoloV3(nn.Module):
    def __init__(self, num_classes: int, num_anchors: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.yolo = nn.Sequential(
            CBA(3, 64, kernel_size=7, stride=2),  # 320 x 320
            BottleneckGroup(64, n_groups=1),
            CBA(64, 128, kernel_size=3, stride=2),  # 160 x 160
            BottleneckGroup(128, n_groups=2),
            CBA(128, 256, kernel_size=3, stride=2),  # 80 x 80
            BottleneckGroup(256, n_groups=8),
            CBA(256, 512, kernel_size=3, stride=2),  # 40 x 40
            BottleneckGroup(512, n_groups=8),
            CBA(512, 1024, kernel_size=3, stride=2),  # 20 x 20
            BottleneckGroup(1024, n_groups=4),
            Objects(1024, self.num_classes, self.num_anchors),
        )

    def forward(self, x):
        return self.yolo(x)
