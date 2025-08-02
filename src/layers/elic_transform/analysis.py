import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import subpel_conv3x3, AttentionBlock
from .conv import conv1x1, conv3x3, conv, deconv
from .res_blk import *


class AnalysisTransformEX(nn.Module):
    def __init__(self, N, M, act=nn.ReLU):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            # TODO
        )

    def forward(self, x):
        x = self.analysis_transform(x)
        return x


class HyperAnalysisEX(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.reduction = nn.Sequential(
            # TODO
        )

    def forward(self, x):
        x = self.reduction(x)
        return x

