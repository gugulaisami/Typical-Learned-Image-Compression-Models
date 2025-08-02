# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.ops import ste_round
from .base import CompressionModel
from ..entropy_models import EntropyBottleneck
from ..entropy_models import GaussianConditional
from compressai.layers import GDN

from .utils import conv, deconv, update_registered_buffers

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class Hyperprior(CompressionModel):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=192, M=320, **kwargs):
        super().__init__(**kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),
        )

        self.h_s = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(N, M, kernel_size=3, stride=1),
        )

        self.N = int(N)
        self.M = int(M)

        self.gaussian_conditional = GaussianConditional(None)
        self.entropy_bottleneck = EntropyBottleneck(self.N)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)
        scales_hat = F.softplus(params)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(abs(y))
        z_hat, z_likelyhoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)
        scales_hat = F.softplus(params)
        y_strings = self.gaussian_conditional.compress(y, scales_hat)

        z_strings = self.entropy_bottleneck.compress(z)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)
        scales_hat = F.softplus(params)
        y_hat = self.gaussian_conditional.decompress(strings[0], scales=scales_hat)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


