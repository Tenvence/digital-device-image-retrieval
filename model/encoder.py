import torch
import torch.nn as nn
from torchvision.models import resnet50


class Encoder(nn.Module):
    def __init__(self, num_classes=3097):
        super(Encoder, self).__init__()
        self.backbone = nn.Sequential(*list(resnet50(pretrained=False).children())[:-1])
        self.fc = nn.Linear(in_features=2048, out_features=2048, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x / torch.norm(x, dim=-1, keepdim=True)  # L2 norm
        return x
