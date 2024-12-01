import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=400, latent_dim=64):
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, input_dim),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        stddev = torch.exp(0.5 * logvar)
        eps = torch.randn_like(stddev)
        return mu + eps * stddev

    def forward(self, x):
        x = x.view(-1, 784)  # MNIST dataset size
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
