import torch, math
import torch.nn as nn

from src.entropy_models import EntropyBottleneck, GaussianConditional
from .utils import update_registered_buffers

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class CompressionModel(nn.Module):
    """Base class for constructing an auto-encoder with at least one entropy
    bottleneck module.

    Args:
        entropy_bottleneck_channels (int): Number of channels of the entropy
            bottleneck
    """

    def __init__(self, init_weights=True):
        super().__init__()
        # self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)

        if init_weights:
            self._initialize_weights()

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, *args):
        raise NotImplementedError()

    def update(self, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        for name, module in self.named_modules():
            if not any(x.startswith(name) for x in state_dict.keys()):
                continue

            if isinstance(module, EntropyBottleneck):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length"],
                    state_dict,
                )

            if isinstance(module, GaussianConditional):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                    state_dict,
                )

        return nn.Module.load_state_dict(self, state_dict, strict=strict)


