import torch
import torch.nn as nn
import unittest

from models.yolo_v1 import YoloV1


def count_conv_layers(model: nn.Module):
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            count += 1
    return count


def count_linear_layers(model: nn.Module):
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            count += 1
    return count


class YoloV1Tests(unittest.TestCase):

    def test_yolo_v1_object(self):

        num_random_images = 10
        num_channels = 3
        img_size = 448
        test_images = torch.rand((num_random_images, num_channels, img_size, img_size))

        num_classes = 9
        yolo_v1 = YoloV1(num_classes=num_classes)
        yolo_v1.eval()
        predictions = yolo_v1(test_images)

        # print(tuple(predictions.shape))

        # (10, 7, 7, 19)
        expected_shape = (num_random_images, yolo_v1.num_cells, yolo_v1.num_cells, 5 * yolo_v1.num_boxes + num_classes)
        self.assertEqual(tuple(predictions.shape), expected_shape)

        # 24 convolutional layers
        self.assertEqual(count_conv_layers(yolo_v1), 24)

        # 2 fully connected layers
        self.assertEqual(count_linear_layers(yolo_v1), 2)


if __name__ == '__main__':
    unittest.main()
