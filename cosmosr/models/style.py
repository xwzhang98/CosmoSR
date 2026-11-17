"""Cosmology-modulated convolutions used by the state56 model."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStyled3d(nn.Module):
    """StyleGAN2 modulation and demodulation with the original initialization."""

    def __init__(
        self, in_chan, out_chan, style_size, kernel_size=3, stride=1, bias=True
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_chan, in_chan, *((kernel_size,) * 3))
        )
        self.stride = stride
        nn.init.kaiming_uniform_(
            self.weight, a=math.sqrt(5), mode="fan_in", nonlinearity="leaky_relu"
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_chan))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        def init_weight(module):
            if type(module) is nn.Linear:
                nn.init.kaiming_uniform_(
                    module.weight,
                    a=math.sqrt(5),
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                if module.bias is not None:
                    nn.init.ones_(module.bias)

        self.style_block = nn.Sequential(nn.Linear(style_size, in_chan))
        self.style_block.apply(init_weight)

    def forward(self, inputs):
        x, style = inputs
        batch, in_chan, *spatial = x.shape
        out_chan, _, *kernel = self.weight.shape
        scales = self.style_block(style).reshape(batch, 1, in_chan, 1, 1, 1)
        weight = self.weight * scales
        weight = weight * torch.rsqrt(
            weight.square().sum(dim=(2, 3, 4, 5), keepdim=True) + 1e-8
        )
        weight = weight.reshape(batch * out_chan, in_chan, *kernel)
        x = x.reshape(1, batch * in_chan, *spatial)
        x = F.conv3d(x, weight, bias=self.bias, stride=self.stride, groups=batch)
        return x.view(batch, out_chan, *x.shape[2:])


class LeakyReLUStyled(nn.LeakyReLU):
    def __init__(self, negative_slope=0.2, inplace=False):
        super().__init__(negative_slope, inplace)

    def forward(self, x, style=None):
        return super().forward(x)


class LeakyReLUStyled2(nn.LeakyReLU):
    def __init__(self, negative_slope=0.2, inplace=False):
        super().__init__(negative_slope, inplace)

    def forward(self, inputs):
        return super().forward(inputs[0])
