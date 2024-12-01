import torch.nn as nn


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=5, padding=2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# noise scheduler
# text encoder
# conditioning mechanism (cross attention layer)
# add noise, remove noise (forward diffusion, reverse diffusion) loss, sampling
