import torch
import torch.nn as nn


class ComputeLoss:
    def __init__(self, model):
        self.device = next(model.parameters()).device
        self.nh = model.nh
        self.nl = model.nl
        self.L1Loss = nn.L1Loss()
    def __call__(self, gaze_pred, labels):
        gloss = torch.zeros(1, device=self.device)
        gloss += self.L1Loss(gaze_pred[..., 0:2], labels.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(gaze_pred[..., 0:2]))

        return gloss, torch.cat((gloss, torch.zeros(1, device=self.device))).detach()
