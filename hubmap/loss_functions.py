import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

EPSILON = 1e-5


class CrossEntropyLoss(torch.nn.Module):
    # from https://github.com/visinf/self-adaptive/blob/master/loss/semantic_seg.py
    def __init__(self, ignore_index: int = 255):

        super(CrossEntropyLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
        self.ignore_index = ignore_index

    def forward(self,
                output: torch.Tensor,
                gt: torch.Tensor):
        """
        Args:
            output: Probabilities for every pixel with stride of 16
            gt: Labeled image at full resolution
        Returns:
            total_loss: Cross entropy loss
        """
        # Compare output and ground-truth at down-sampled resolution
        gt = gt.long().squeeze(1)
        loss = self.criterion(output, gt)

        # Compute total loss
        total_loss = (loss[gt != self.ignore_index]).mean()

        return total_loss


class DiceCELoss(nn.Module):
    def __init__(self, ignore_index: int = -1):
        """
        Only for binary segmentation.
        We implement the "batch Dice" like in nnU-Net
        :param reduction: str. Mode to merge the batch of sample-wise loss values.
        :param squared: bool. If True, squared the terms in the denominator of the Dice loss.
        """
        super(DiceCELoss, self).__init__()
        self.ce = CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def _prepare_data(self, input_batch, target):
        num_out_classes = 2
        # Prepare the batch prediction
        flat_input = torch.reshape(input_batch, (input_batch.size(0), input_batch.size(1), -1))  # b,c,s
        flat_target = torch.reshape(target, (target.size(0), -1))  # b,s

        # Compute softmax and one-hot proba - keep only foreground class
        pred_proba = F.softmax(flat_input, dim=1)  # b,c,s
        pred_proba = pred_proba[:, 1, :]  # b,s
        # Remove the pixels to ignore
        pred_proba = pred_proba[flat_target != self.ignore_index]  # bxs
        flat_target = flat_target[flat_target != self.ignore_index]  # bxs
        target_proba = F.one_hot(flat_target, num_classes=num_out_classes)  # bxs, c
        target_proba = target_proba.float()[:, 1]  # bxs

        return pred_proba, target_proba

    def forward(self, input_batch, target):
        ce = self.ce(input_batch, target)
        pred_proba, target_proba = self._prepare_data(input_batch, target)

        # Compute the dice for the foreground class
        num = pred_proba * target_proba  # bxs, --p*g
        num = torch.sum(num, dim=0)  # 1,
        den1 = torch.sum(pred_proba, dim=0)  # 1,
        den2 = torch.sum(target_proba, dim=0)  # 1,
        dice_loss = 1. - (2. * num) / (den1 + den2 + EPSILON)

        loss = ce + dice_loss

        return loss
