""" Loss functions. """

import torch
import torch.nn as nn

from models.model import XModel
from utils.general import angular_error


class ComputeLoss:

    # Compute losses
    def __init__(self, model):
        self.device = next(model.parameters()).device
        self.nh = model.nh
        self.nl = model.nl
        # Define criteria
        self.MSELoss = nn.MSELoss().to(self.device)
        self.L1Loss = nn.L1Loss()
        self.BCEangular = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.91, device=self.device))  # 二元交叉熵损失

    def __call__(self, gaze_pred, labels):  # 预测标签和真实标签作为参数
        sight = 1.0
        # angular = 0.5  # 更改
        angular = 0.5
        gloss, aloss = torch.zeros(1, device=self.device), \
                              torch.zeros(1, device=self.device)
        # Sight regression loss
        gloss += self.L1Loss(gaze_pred[..., 0:2], labels.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(gaze_pred[..., 0:2]))
        # Angular error loss
        angular_error_matrix = angular_error(gaze_pred[..., 0:2], labels)
        aloss += angular_error_matrix.mean()  # 当前批次的平均角度误差
        gloss *= sight
        aloss *= angular
        return gloss + aloss, torch.cat((gloss, aloss)).detach()


if __name__ == "__main__":
    pred = (torch.rand(16, 1, 1, 1, 2) - 0.5) * 84 - 42  # 随机初始化在 [-42, 42] 范围内
    targets = (torch.rand(16, 2) - 0.5) * 84 - 42  # 随机初始化在 [-42, 42] 范围内
    model = XModel("../models/yaml/resnet50-FP-c2f.yaml")
    computeloss = ComputeLoss(model)
    loss = computeloss(pred, targets)
    print(loss)
