import torch

from models.bottleneck import BottleneckYoloV3, BottleneckResnet, BottleneckGroup


def test_bottleneck_yolo_v3():

    # arrange
    batch_size = 4
    in_out_channels = 64
    img_size = 80
    images = torch.rand((batch_size, in_out_channels, img_size, img_size))

    # act, groups=1
    bottleneck = BottleneckYoloV3(in_out_channels, ratio=1.0)
    output = bottleneck(images)

    # assert
    assert output.shape == (batch_size, in_out_channels, img_size, img_size)


def test_bottleneck_resnet():

    # arrange
    batch_size = 4
    in_out_channels = 64
    img_size = 80
    images = torch.rand((batch_size, in_out_channels, img_size, img_size))

    # act, groups=1
    bottleneck = BottleneckResnet(in_out_channels, ratio=1.0)
    output = bottleneck(images)

    # assert
    assert output.shape == (batch_size, in_out_channels, img_size, img_size)


def test_bottleneck_group():

    # arrange
    batch_size = 4
    in_out_channels = 64
    img_size = 80
    images = torch.rand((batch_size, in_out_channels, img_size, img_size))

    # act, groups=1
    bottleneck = BottleneckGroup(in_out_channels, n_groups=1)
    output = bottleneck(images)

    # assert
    assert output.shape == (batch_size, in_out_channels, img_size, img_size)

    # act, groups=1
    bottleneck = BottleneckGroup(in_out_channels, n_groups=2)
    output = bottleneck(images)

    # assert
    assert output.shape == (batch_size, in_out_channels, img_size, img_size)
