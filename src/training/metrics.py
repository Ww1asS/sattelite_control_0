import torch
from torchmetrics import JaccardIndex, F1Score

#Расчет метрик IoU и F1
class ChangeDetectionMetrics:

    def __init__(self, threshold=0.5):
        self.iou = JaccardIndex(task="binary", threshold=threshold)
        self.f1 = F1Score(task="binary", threshold=threshold)

    def to(self, device):
        self.iou = self.iou.to(device)
        self.f1 = self.f1.to(device)
        return self

    def update(self, logits, target):
        probs = torch.sigmoid(logits)
        self.iou.update(probs, target.int())
        self.f1.update(probs, target.int())

    def compute(self):
        return {
            "iou": self.iou.compute().item(),
            "f1": self.f1.compute().item(),
        }

    def reset(self):
        self.iou.reset()
        self.f1.reset()