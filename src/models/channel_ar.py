import torch
import math
import torch.nn.functional as F
import numpy as np
from .utils import conv, update_registered_buffers, deconv

import torch.nn as nn
from compressai.ans import BufferedRansEncoder, RansDecoder
from src.entropy_models import EntropyBottleneck, GaussianConditional
from src.layers import conv3x3, subpel_conv3x3, GDN
from src.ops import ste_round
from .base import CompressionModel

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class Channel_AR(CompressionModel):
    def __init__(self, N=192, M=320, num_slices=5, **kwargs):
        super().__init__()

        self.num_slices = num_slices
        self.slice_depth = M // num_slices
        self.max_support_slices = self.num_slices // 2

        self.g_a = nn.Sequential(
            # TODO
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            # TODO
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            # TODO
            conv(M, M, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(M, 256, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(256, N, kernel_size=5, stride=2),
        )

        self.h_mean_s = nn.Sequential(
            # TODO
            deconv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(N, 256, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(256, M, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.h_scale_s = nn.Sequential(
            # TODO
            deconv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(N, 256, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(256, M, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.cc_mean_transforms = nn.ModuleList(
            deconv(M, 224, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            deconv(224, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            deconv(128, 32, kernel_size=3, stride=1),
        )
        self.cc_scale_transforms = nn.ModuleList(
            deconv(M, 224, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            deconv(224, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            deconv(128, 32, kernel_size=3, stride=1),
            nn.exp(inplace=True)
        )
        self.lrp_transforms = nn.ModuleList(
            deconv(M, 224, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            deconv(224, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            deconv(128, 32, kernel_size=3, stride=1),
        )
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)

            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)

            indexes = self.gaussian_conditional.build_indexes(scale)
            y_string = self.gaussian_conditional.compress(y_slice, indexes, mu)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        y_strings, z_strings = strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_hat_slices = []

        for slice_index, y_string in enumerate(y_strings):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)

            indexes = self.gaussian_conditional.build_indexes(scale)
            y_hat_slice = self.gaussian_conditional.decompress(y_string, indexes, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}
