import torch.nn as nn
import torch.nn.functional as F
import torch


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        n_classes = inputs.size(-1)
        one_hot = torch.zeros_like(log_probs).scatter(1, targets.unsqueeze(1), 1)
        targets = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_classes - 1)
        loss = (-targets * log_probs).sum(dim=-1).mean()
        return loss