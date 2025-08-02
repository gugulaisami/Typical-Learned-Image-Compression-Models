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
import math, argparse, warnings, os
import torch
import torch.nn as nn

# # import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from einops import rearrange
import gc



def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)


def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)


def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",     
            state_dict,
            policy,
            dtype,
        )


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):     # SN -1 + k - 2p
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Compression')
    # Base configure
    parser.add_argument("--channels", type=int, default=128, help="Channels in Main Auto-encoder.")
    parser.add_argument("--last_channels", type=int, default=128, help="Channels of compression feature.")
    parser.add_argument("--hyper_channels", type=int, default=128, help="Channels of hyperprior feature.")
    parser.add_argument("--num_parameter", type=int, default=2,
                        help="distribution parameter num: 1 for sigma, 2 for mean&sigma, 3 for mean&sigma&pi")

    # Configure for Transfomer Entropy Model
    parser.add_argument("--dim_embed", type=int, default=384, help="Dimension of transformer embedding.")
    parser.add_argument("--depth", type=int, default=6, help="Depth of CiT.")
    parser.add_argument("--heads", type=int, default=6, help="Number of transformer head.")
    parser.add_argument("--mlp_ratio", type=int, default=4, help="Ratio of transformer MLP.")
    parser.add_argument("--dim_head", type=int, default=64, help="Dimension of transformer head.")
    parser.add_argument("--trans_no_norm", dest="trans_norm", action="store_false", default=True, help="Use LN in transformer.")
    parser.add_argument("--dropout", type=float, default=0., help="Dropout ratio.")
    parser.add_argument("--position_num", type=int, default=6, help="Position information num.")
    parser.add_argument("--att_noscale", dest="att_scale", action="store_false", default=True, help="Use Scale in Attention.")
    parser.add_argument("--no_rpe_shared", dest="rpe_shared", action="store_false", default=True, help="Position Shared in layers.")
    parser.add_argument("--scale", type=int, default=2, help="Downscale of hyperprior of CiT.")
    parser.add_argument("--mask_ratio", type=float, default=0., help="Pretrain model: mask ratio.")
    parser.add_argument("--attn_topk", type=int, default=-1, help="Top K filter for Self-attention.")    
    parser.add_argument("--grad_norm_clip", type=float, default=0., help="grad_norm_clip.")

    return parser

