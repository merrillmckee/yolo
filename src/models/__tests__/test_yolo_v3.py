import torch

from models.yolo_v3 import YoloV3, Objects


def test_objects():

    # arrange
    batch_size = 4
    in_channels = 128
    img_size = 20
    num_classes = 9
    num_anchors = 3
    images = torch.rand((batch_size, in_channels, img_size, img_size))

    # act
    objects = Objects(in_channels, num_classes, num_anchors)
    output = objects(images)

    # assert (4, 3, 20, 20, 14)
    assert output.shape == (batch_size, num_anchors, 20, 20, 5 + num_classes)


def test_yolo_v3():

    # arrange
    batch_size = 4
    in_channels = 3
    img_size = 640
    num_classes = 9
    num_anchors = 3
    images = torch.rand((batch_size, in_channels, img_size, img_size))

    # act
    yolo = YoloV3(num_classes, num_anchors)
    output = yolo(images)

    # assert (4, 3, 20, 20, 14)
    assert output.shape == (batch_size, num_anchors, 20, 20, 5 + num_classes)
