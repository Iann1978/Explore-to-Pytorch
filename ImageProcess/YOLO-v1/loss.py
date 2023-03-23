# https://blog.csdn.net/wjytbest/article/details/116116966

import torch
from torch import nn as nn
from torch import tensor
import torchvision.ops as ops
import numpy as np


class YoloLoss_v1(nn.Module):
    def __init__(self, batch_size: int = 16, num_classes: int = 90, S: int = 7, B: int = 2):
        super().__init__()
        self.batch_size = batch_size
        self.S = S
        self.B = B
        self.image_size = (512,512)
        self.num_classes = num_classes


    def forward(self, pred, target):

        confidences = torch.zeros(self.batch_size, self.S, self.S, self.B)
        classes = torch.zeros(self.batch_size, self.S, self.S, self.B, dtype=int)
        boxes = torch.zeros(self.batch_size, self.S, self.S, self.B, 4)
        for bi in range(self.batch_size):
            for t in target[bi]:
                for obj in t:
                    bbox = t['bbox']
                    ix = int(min(bbox[0]/self.image_size[0] * 7, 6))
                    iy = int(min(bbox[1]/self.image_size[1] * 7, 6))
                    ib = int(np.clip(bbox[2]/bbox[3], 0.0, 1.0))
                    confidences[bi, iy, ix, ib] = 1
                    classes[bi, iy, ix, ib] = t['category_id']-1
                    boxes[bi, iy, ix, ib] = tensor(t['bbox'])

        max_num_of_classes = classes.max()

        # Calculate classes losses
        class_true = torch.nn.functional.one_hot(torch.tensor(classes), num_classes=self.num_classes)
        class_pred = pred[:, 10:, :, :].permute(0, 3, 2, 1)
        class0_loss = ((class_pred-class_true[:, :, :, 0, :]) ** 2).sum(dim=-1)
        class1_loss = ((class_pred-class_true[:, :, :, 1, :]) ** 2).sum(dim=-1)
        class_loss = torch.stack((class0_loss,class1_loss), dim=-1)
        class_loss = confidences * class_loss

        # Calculate confidence losses
        conf_true = confidences.clone()
        conf_pred = pred[:, [4, 9], :, :].permute(0, 3, 2, 1)
        conf_loss = confidences * (conf_pred - conf_true) ** 2 + 0.5 * (1-confidences) * (conf_pred - conf_true) ** 2

        # Calculate box losses

        boxes_true = boxes
        boxes_true[:, :, :, :, [2, 3]].sqrt_()
        boxes0_pred = pred[:, [0, 1, 2, 3], :, :].permute(0, 3, 2, 1)
        boxes1_pred = pred[:, [5, 6, 7, 8], :, :].permute(0, 3, 2, 1)
        boxes_pred = torch.stack((boxes0_pred, boxes1_pred), dim=-2)
        boxes_pred[:, :, :, :, [2, 3]].sqrt_()
        boxes_loss = ((boxes_pred - boxes_true)**2).sum(dim=-1)
        boxes_loss = confidences*boxes_loss


        loss = conf_loss.sum() + class_loss.sum() + boxes_loss.sum()
        return loss

