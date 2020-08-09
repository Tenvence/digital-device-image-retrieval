import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision.models import resnet50


class Encoder(nn.Module):
    def __init__(self, num_classes=3097, contrast_dim=16):
        super(Encoder, self).__init__()
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-1])
        self.contrast_fc = nn.Linear(in_features=2048, out_features=contrast_dim, bias=False)
        self.cla_fc = nn.Linear(in_features=2048, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)

        contrast_output = self.contrast_fc(x)
        contrast_output = f.tanh(contrast_output)  # 将数据向 +1/-1 方向压缩, 使特征向量向特征空间中某个象限的中间位置回归
        contrast_output = contrast_output / torch.norm(contrast_output, dim=-1, keepdim=True)  # L2 norm

        cla_output = self.cla_fc(x)
        cla_output = f.softmax(cla_output, dim=-1)

        return contrast_output, cla_output
