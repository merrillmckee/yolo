import torch

from models.cba import CBA


def test_cba():

    # arrange
    batch_size = 4
    in_channels = 3
    out_channels = 64
    kernel_size = 7
    img_size = 80
    images = torch.rand((batch_size, in_channels, img_size, img_size))

    # act, stride==1
    cba = CBA(in_channels, out_channels, kernel_size)  # stride==1
    output = cba(images)

    # assert
    assert output.shape == (batch_size, out_channels, img_size, img_size)

    # act, stride==2
    cba = CBA(in_channels, out_channels, kernel_size, stride=2)
    output = cba(images)

    # assert
    assert output.shape == (batch_size, out_channels, img_size // 2, img_size // 2)
