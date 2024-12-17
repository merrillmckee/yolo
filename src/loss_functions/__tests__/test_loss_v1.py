import torch
import unittest

# from loss_functions.loss_v1 import LossV1
from torch import Tensor

from loss_functions.loss_v1 import Loss
from torchvision.ops import box_iou


class LossV1Tests(unittest.TestCase):
    # def test_compute_iou(self):
    #     loss = LossV1(num_classes=9)
    #     n = 2
    #     b = loss.b
    #     c = loss.c
    #     s = loss.s
    #
    #     # (N, S, S, B * 5 + C)
    #     predictions = torch.zeros((n, s, s, b * 5 + c))
    #     targets = torch.zeros((n, s, s, b * 5 + c))
    #     ious = loss.compute_iou(predictions, targets)
    #
    #
    #     self.assertEqual(True, False)  # add assertion here

    def test_compute_iou_2(self):

        loss = Loss()
        boxes_1 = torch.zeros((3, 4))
        boxes_2 = torch.zeros((2, 4))
        boxes_1[0, :] = torch.tensor([1, 2, 4, 6])  # area 12
        boxes_2[0, :] = torch.tensor([2, 3, 6, 8])  # area 20  # intersection: 6 and union: 26 so iou = 6/26


        ious_1 = loss.compute_iou(boxes_1, boxes_2)

        self.assertEqual(ious_1.shape, (3, 2))

        ious_2 = box_iou(boxes_1, boxes_2)

        self.assertEqual(ious_2.shape, (3, 2))

        print()

    # def test_forward(self):
    #
    #     s = 7
    #     n = 2
    #     b = 2
    #     c = 9
    #     loss_fn = Loss(feature_size=7, num_bboxes=2, num_classes=9, lambda_coord=5.0, lambda_noobj=0.5)
    #
    #     predictions = torch.ones((n, s, s, b * 5 + c))
    #     targets = torch.zeros((n, s, s, b * 5 + c)) + 0.5
    #     loss = loss_fn(predictions, targets)
    #
    #     self.assertEqual(loss.shape, (1, ))


if __name__ == '__main__':
    unittest.main()
