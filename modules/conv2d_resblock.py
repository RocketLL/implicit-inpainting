from torch import nn
from torch.nn.functional import relu
from functools import partial


class Conv2DResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
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

        self.block = nn.Sequential(
            Conv2d(),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            Conv2d(),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        res = x
        return relu(self.block(x) + res)
