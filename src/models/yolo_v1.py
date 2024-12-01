import torch.nn as nn


class YoloV1(nn.Module):
    def __init__(self, num_classes: int, num_cells: int = 7, num_boxes: int = 2):
        super().__init__()

        self.num_classes = num_classes  # variable C in paper
        self.num_cells = num_cells      # variable S in paper
        self.num_boxes = num_boxes      # variable B in paper

        # as described in the YoloV1 paper: https://arxiv.org/pdf/1506.02640
        self.model = nn.Sequential(
            # 448x448 images
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(num_features=64), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),

            # 112x112 images
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=192), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),

            # 56x56 images
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=128), nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=256), nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=256), nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=512), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),

            # 28x28 images
            #   x4
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=256), nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=512), nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=256), nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=512), nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=256), nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=512), nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=256), nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=512), nn.LeakyReLU(0.1),
            #
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=512), nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=1024), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),

            # 14x14 images
            #   x2
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=512), nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=1024), nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=512), nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=1024), nn.LeakyReLU(0.1),
            #
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=1024), nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(num_features=1024), nn.LeakyReLU(0.1),

            # 7x7 images
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=1024), nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding="same", bias=False), nn.BatchNorm2d(num_features=1024), nn.LeakyReLU(0.1),

            # FC: Flatten
            nn.Flatten(),
            nn.Linear(in_features=7 * 7 * 1024, out_features=4096), nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(
                in_features=4096,
                out_features=num_cells * num_cells * (5 * num_boxes + num_classes),  # 5: w, h, x, y, p_object
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.num_cells, self.num_cells, 5 * self.num_boxes + self.num_classes)
        return x
