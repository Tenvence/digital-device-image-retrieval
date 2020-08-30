import torch
import torch.nn as nn
import torchvision.models as cv_models
from torch.cuda.amp import autocast


class Encoder(nn.Module):
    def __init__(self, feature_num):
        super(Encoder, self).__init__()
        self.backbone = nn.Sequential(*list(cv_models.resnet50(pretrained=True).children())[:-1])
        self.mlp = nn.Sequential(
            nn.Linear(in_features=2048, out_features=feature_num),
            nn.BatchNorm1d(num_features=feature_num),
            nn.ReLU(),
            nn.Linear(in_features=feature_num, out_features=feature_num)
        )

    @autocast()
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)

        return x

#
# class EmbeddedFeatureWrapper(nn.Module):
#     """
#     DNN head: pools, down-projects, and normalizes DNN features to be of unit length.
#     """
#
#     def __init__(self, input_dim, output_dim, dropout=0):
#         super(EmbeddedFeatureWrapper, self).__init__()
#         self.output_dim = output_dim
#         self.dropout = dropout
#         if output_dim != 4096:
#             self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         self.standardize = nn.LayerNorm(input_dim, elementwise_affine=False)
#         self.remap = None
#         if input_dim != output_dim:
#             self.remap = nn.Linear(input_dim, output_dim, bias=False)
#         if dropout > 0:
#             self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         if self.output_dim != 4096:
#             x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.standardize(x)
#         if self.remap:
#             x = self.remap(x)
#         if self.dropout > 0:
#             x = self.dropout(x)
#         x = nn.functional.normalize(x, dim=1)
#         return x
#
#
# class L2NormalizedLinearLayer(nn.Module):
#     """
#     Apply a linear layer to the input, where the weights are normalized to be of unit length.
#     """
#
#     def __init__(self, input_dim, output_dim):
#         super(L2NormalizedLinearLayer, self).__init__()
#         self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         # Initialization from nn.Linear (https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/modules/linear.py#L129)
#
#     def forward(self, x):
#         norm_weight = nn.functional.normalize(self.weight, dim=1)
#         prediction_logits = nn.functional.linear(x, norm_weight)
#         return prediction_logits
#
#
# class NormSoftmaxLoss(nn.Module):
#     """
#     Apply temperature scaling on logits before computing the cross-entropy loss.
#     """
#
#     def __init__(self, temperature=0.05):
#         super(NormSoftmaxLoss, self).__init__()
#         self.temperature = temperature
#         self.loss_fn = nn.CrossEntropyLoss()
#
#     def forward(self, prediction_logits, instance_targets):
#         loss = self.loss_fn(prediction_logits / self.temperature, instance_targets)
#         return loss
