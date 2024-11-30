from torch import nn
from torch.utils.data import DataLoader
import torch
import torchvision


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        # define upsample & downsample layers
        # padding = (kernel - 1) / 2
        self.down_layers = nn.ModuleList(
            [
                nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
            ]
        )
        self.up_layers = nn.ModuleList(
            [
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
            ]
        )
        self.down_sample = nn.MaxPool2d(2)
        self.up_sample = nn.Upsample(scale_factor=2)
        self.act_func = nn.SELU()

    def forward(self, x):
        skip_connections = []
        for i, dl in enumerate(self.down_layers):
            x = self.act_func(dl(x))
            if i < len(self.down_layers) - 1:
                skip_connections.append(x)
                x = self.down_sample(x)

        for i, ul in enumerate(self.up_layers):
            x = self.act_func(ul(x))
            if i < len(self.up_layers) - 1:
                x = self.up_sample(x)
                x += skip_connections.pop()
        # for i, l in enumerate(self.up_layers):
        #     if i > 0:  # For all except the first up layer
        #         x = self.up_sample(x)  # Upscale
        #         x += skip_connections.pop()  # Fetching stored output (skip connection)
        #     x = self.act_func(l(x))
        return x


# unet = SimpleUNet()
# x = torch.rand(8, 1, 28, 28)
# out = unet.forward(x)
# print(out.shape)
# # for param in unet.parameters():
# #     print(param.numel)
# print(sum([p.numel() for p in unet.parameters()]))
# print(unet.parameters())
