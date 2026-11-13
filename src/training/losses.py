import torch.nn as nn
import segmentation_models_pytorch as smp

#Loss функция BCE + Dice
class CombinedLoss(nn.Module):

    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.dice_loss = smp.losses.DiceLoss(mode="binary")
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits, target):
        return (
            self.bce_weight * self.bce_loss(logits, target)
            + self.dice_weight * self.dice_loss(logits, target)
        )