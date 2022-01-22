import torch
from torch import nn
from torch.nn.functional import relu, softplus
from torch.distributions import MultivariateNormal
from functools import partial

"""
Convolutional Conditional Neural Processes,
    https://arxiv.org/abs/1910.13556

Implementation based on
    https://github.com/makora9143/pytorch-convcnp
"""


class Conv2DResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super().__init__()

        Conv2d = partial(
            nn.Conv2d,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
        )

        self.conv2ds = nn.Sequential(
            Conv2d(),
            nn.ReLU(),
            Conv2d(),
            nn.ReLU(),
        )

    def forward(self, x):
        return relu(self.conv2ds(x) + x)


class Conv2dCNP(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.channels = channels
        self.conv = nn.Conv2d(channels, 128, 9, 1, 4)

        self.nn = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 1, 1, 0),
            Conv2DResBlock(128, 128),
            Conv2DResBlock(128, 128),
            Conv2DResBlock(128, 128),
            Conv2DResBlock(128, 128),
            nn.Conv2d(128, 2 * channels, 1, 1, 0),
        )

    def forward(self, image, mask):
        image_prime = self.conv(image)
        mask_prime = self.conv(mask)

        x = torch.cat((image_prime, mask_prime), 1)
        phi = self.nn(x)
        mean, std = phi.split(self.channels, 1)
        std = softplus(std)

        return MultivariateNormal(mean, scale_tril=std.diag_embed())
