"""Cosmology-conditioned generator and discriminator used by state56."""

from math import log2

import torch
import torch.nn as nn

from .narrow import narrow_by
from .resample import Resampler, Resampler2
from .style import ConvStyled3d, LeakyReLUStyled
from .styled_conv import ResStyledBlock


class G(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        style_size,
        scale_factor=2,
        chan_base=512,
        chan_min=64,
        chan_max=512,
    ):
        super().__init__()
        self.style_size = style_size
        self.scale_factor = scale_factor
        num_blocks = round(log2(scale_factor))
        assert chan_min <= chan_max

        def chan(block):
            return min(max(chan_base >> block, chan_min), chan_max)

        self.block0 = nn.Sequential(
            ConvStyled3d(in_chan, chan(0), style_size, 1),
            LeakyReLUStyled(0.2, True),
        )
        self.blocks = nn.ModuleList(
            [
                HBlock(chan(block), chan(block + 1), out_chan, style_size)
                for block in range(num_blocks)
            ]
        )

    def forward(self, x, style):
        skip = x
        x = self.block0((x, style))
        for block in self.blocks:
            x, skip, style = block(x, skip, style)
        return skip


class HBlock(nn.Module):
    """Upsample both branches; valid convolutions give output side 2*N-6."""

    def __init__(self, prev_chan, next_chan, out_chan, style_size):
        super().__init__()
        self.upsample = Resampler(3, 2)
        self.noise_upsample = nn.Sequential(
            AddNoise(prev_chan),
            self.upsample,
        )
        self.conv = nn.Sequential(
            ConvStyled3d(prev_chan, next_chan, style_size, 3),
            LeakyReLUStyled(0.2, True),
        )
        self.addnoise = AddNoise(next_chan)
        self.conv1 = nn.Sequential(
            ConvStyled3d(next_chan, next_chan, style_size, 3),
            LeakyReLUStyled(0.2, True),
        )
        # The checkpoint includes demodulation and activation in this projection.
        self.proj = nn.Sequential(
            ConvStyled3d(next_chan, out_chan, style_size, 1),
            LeakyReLUStyled(0.2, True),
        )

    def forward(self, x, skip, style):
        x = self.noise_upsample(x)
        x = self.conv((x, style))
        x = self.addnoise(x)
        x = self.conv1((x, style))
        skip = narrow_by(self.upsample(skip), 2)
        skip = skip + self.proj((x, style))
        return x, skip, style


class AddNoise(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.std = nn.Parameter(torch.zeros([chan]))

    def forward(self, x):
        noise = torch.randn_like(x[:, :1])
        shape = (-1,) + (1,) * (x.dim() - 2)
        return x + self.std.view(shape) * noise


class D(nn.Module):
    """Input channels: upsampled LR (6), CIC density (8), SR or HR (6)."""

    def __init__(
        self,
        in_chan,
        out_chan,
        style_size,
        scale_factor=2,
        chan_base=512,
        chan_min=64,
        chan_max=512,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.style_size = style_size
        num_blocks = round(log2(scale_factor))
        assert chan_min <= chan_max

        def chan(block):
            value = chan_base >> block if block >= 0 else chan_base << -block
            return min(max(value, chan_min), chan_max)

        self.block0 = nn.Sequential(
            ConvStyled3d(in_chan + 8, chan(num_blocks), style_size, 1),
            LeakyReLUStyled(0.2, True),
        )
        self.blocks = nn.ModuleList()
        for block in reversed(range(num_blocks)):
            self.blocks.append(
                ResStyledBlock(
                    chan(block + 1),
                    chan(block),
                    style_size,
                )
            )
            self.blocks.append(Resampler2(3, 0.5))

        self.block9 = nn.Sequential(
            ConvStyled3d(chan(0), chan(-1), style_size, 1),
            LeakyReLUStyled(0.2, True),
        )
        self.block10 = ConvStyled3d(chan(-1), 1, style_size, 1)

    def forward(self, x, style):
        x = self.block0((x, style))
        for block in self.blocks:
            x = block((x, style))
        x = self.block9((x, style))
        return self.block10((x, style))
