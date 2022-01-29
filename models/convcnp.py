import torch
from torch import nn
from torch.nn.functional import softplus
from torch.distributions import MultivariateNormal
from modules import Conv2DResBlock

"""
Convolutional Conditional Neural Processes,
    https://arxiv.org/abs/1910.13556

Implementation based on
    https://github.com/makora9143/pytorch-convcnp
"""


class Conv2dCNP(nn.Module):
    def __init__(self, dims=3, channels=128, blocks=8):
        super().__init__()

        self.dims = dims
        self.conv = nn.Conv2d(dims, channels, 9, 1, 4)

        self.nn = nn.Sequential(
            nn.Conv2d(channels + channels, channels, 1, 1, 0),
            *[
                Conv2DResBlock(channels, channels, kernel_size=5, stride=1, padding=2)
                for _ in range(blocks)
            ],
            nn.Conv2d(channels, 2 * dims, 1, 1, 0),
        )

    def forward(self, image, mask):
        image_prime = self.conv(image)
        mask_prime = self.conv(mask)

        x = torch.cat((image_prime, mask_prime), 1)
        phi = self.nn(x)
        mean, std = phi.split(self.dims, 1)
        std = softplus(std)

        return MultivariateNormal(mean, scale_tril=std.diag_embed())
