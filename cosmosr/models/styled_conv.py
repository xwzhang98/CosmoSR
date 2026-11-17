"""Residual discriminator block, with checkpoint-compatible parameter names."""

import torch.nn as nn

from .narrow import narrow_like
from .style import ConvStyled3d, LeakyReLUStyled2


class ResStyledBlock(nn.Module):
    def __init__(self, in_chan, out_chan, style_size):
        super().__init__()
        mid_chan = min(in_chan, out_chan)
        self.convs = nn.ModuleList(
            [
                ConvStyled3d(in_chan, mid_chan, style_size, 3),
                LeakyReLUStyled2(),
                ConvStyled3d(mid_chan, out_chan, style_size, 3),
                LeakyReLUStyled2(),
            ]
        )
        self.skip = ConvStyled3d(in_chan, out_chan, style_size, 1)

    def forward(self, inputs):
        x, style = inputs
        skip = self.skip((x, style))
        for layer in self.convs:
            x = layer((x, style))
        x += narrow_like(skip, x)
        return x
