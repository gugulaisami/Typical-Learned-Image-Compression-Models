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

import struct
from typing import List, Dict, Any
from pathlib import Path

import torch
# from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

def consume_prefix_in_state_dict_if_present(
    state_dict: Dict[str, Any], prefix: str
) -> None:
    r"""Strip the prefix in state_dict in place, if any.

    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.

    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) :]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)

def get_padding_size(height, width, p=64):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    # padding_left = (new_w - width) // 2
    padding_left = 0
    padding_right = new_w - width - padding_left
    # padding_top = (new_h - height) // 2
    padding_top = 0
    padding_bottom = new_h - height - padding_top
    return padding_left, padding_right, padding_top, padding_bottom


def get_downsampled_shape(height, width, p):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    return int(new_h / p + 0.5), int(new_w / p + 0.5)


def get_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if "state_dict" in ckpt:
        ckpt = ckpt['state_dict']
    if "net" in ckpt:
        ckpt = ckpt["net"]
    consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
    return ckpt

def encode_size(height, width, output):
    with Path(output).open("wb") as f:
        write_uints(f, (height, width))

def encode_head(quality, height, width, output):
    with Path(output).open("wb") as f:
        write_uints(f, (quality, height, width))

def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def write_ushorts(fd, values, fmt=">{:d}H"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_ushorts(fd, n, fmt=">{:d}H"):
    sz = struct.calcsize("H")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


# def encode_i(height, width, q_index, bit_stream, output):
#     with Path(output).open("wb") as f:
#         stream_length = len(bit_stream)

#         write_uints(f, (height, width))
#         write_ushorts(f, (q_index,))
#         write_uints(f, (stream_length,))
#         write_bytes(f, bit_stream)

def encode_i(height, width, bit_stream, output):
    with Path(output).open("wb") as f:
        stream_length = len(bit_stream)

        write_uints(f, (height, width))
        write_uints(f, (stream_length,))
        write_bytes(f, bit_stream)

def encode_i_skip(height, width, q_index, bit_stream, output, threshold):
    print(threshold)
    threshold = [int(item * 10000) for item in threshold]
    print(threshold)
    with Path(output).open("wb") as f:
        stream_length = len(bit_stream)

        write_uints(f, (height, width))
        write_ushorts(f, (q_index,))
        write_ushorts(f, (threshold[0],))
        write_ushorts(f, (threshold[1],))
        write_ushorts(f, (threshold[2],))
        write_ushorts(f, (threshold[3],))
        write_uints(f, (stream_length,))
        write_bytes(f, bit_stream)

# def decode_i(inputpath):
#     with Path(inputpath).open("rb") as f:
#         header = read_uints(f, 2)
#         height = header[0]
#         width = header[1]
#         q_index = read_ushorts(f, 1)[0]
#         stream_length = read_uints(f, 1)[0]

#         bit_stream = read_bytes(f, stream_length)

#     return height, width, q_index, bit_stream

def decode_i(inputpath):
    with Path(inputpath).open("rb") as f:
        header = read_uints(f, 2)
        height = header[0]
        width = header[1]
        stream_length = read_uints(f, 1)[0]

        bit_stream = read_bytes(f, stream_length)

    return height, width, bit_stream

def decode_i_skip(inputpath):
    threshold = []
    with Path(inputpath).open("rb") as f:
        header = read_uints(f, 2)
        height = header[0]
        width = header[1]
        q_index = read_ushorts(f, 1)[0]
        threshold.append(read_ushorts(f, 1)[0]/10000)
        threshold.append(read_ushorts(f, 1)[0]/10000)
        threshold.append(read_ushorts(f, 1)[0]/10000)
        threshold.append(read_ushorts(f, 1)[0]/10000)
        stream_length = read_uints(f, 1)[0]

        bit_stream = read_bytes(f, stream_length)

    print(threshold)

    return height, width, q_index, bit_stream, threshold

def decode_size(inputpath):
    with Path(inputpath).open("rb") as f:
        header = read_uints(f, 2)
        height = header[0]
        width = header[1]

    return height, width

def decode_head(inputpath):
    with Path(inputpath).open("rb") as f:
        header = read_uints(f, 3)
        quality = header[0]
        height = header[1]
        width = header[2]

    return quality, height, width

def encode_p(string, q_in_ckpt, q_index, output):
    with Path(output).open("wb") as f:
        string_length = len(string)
        write_uchars(f, ((q_in_ckpt << 7) + (q_index << 1),))
        write_uints(f, (string_length,))
        write_bytes(f, string)


def decode_p(inputpath):
    with Path(inputpath).open("rb") as f:
        flag = read_uchars(f, 1)[0]
        q_in_ckpt = (flag >> 7) > 0
        q_index = ((flag & 0x7f) >> 1)

        header = read_uints(f, 1)
        string_length = header[0]
        string = read_bytes(f, string_length)

    return q_in_ckpt, q_index, string
