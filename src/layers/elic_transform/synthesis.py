import torch.nn as nn
from src.layers import AttentionBlock
from .conv import deconv
from .res_blk import ResidualBottleneck


class SynthesisTransformEX(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            # TODO
        )

    def forward(self, x):
        x = self.synthesis_transform(x)
        return x


class HyperSynthesisEX(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.increase = nn.Sequential(
            # TODO
        )

    def forward(self, x):
        x = self.increase(x)
        return x
