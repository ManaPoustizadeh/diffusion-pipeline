import torch
from torch.utils.data import DataLoader

from diffusion.components.scheduler import DiffusionScheduler
from diffusion.components.unet import UNet
from diffusion.components.vae import VAE


def train(
    net: UNet,
    scheduler: DiffusionScheduler,
    vae: VAE,
    optimizer,
    train_dataloader: DataLoader,
    epochs: int = 3,
):
    vae.eval()
    net.train()

    loss_func = torch.nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0
        for x_0 in train_dataloader:
            z_0 = vae.encoder(x_0)

            t = torch.randint(0, scheduler.timesteps, (x_0.size(0),))
            x_t, noise = scheduler.forward_diffusion(z_0, t)
            predicted_noise = net(x_t)

            loss = loss_func(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch}, loss: {epoch_loss/len(train_dataloader):.4f}")


@torch.no_grad()
def infer(
    net: UNet, scheduler: DiffusionScheduler, vae: VAE, input_dim: int, timesteps: int
):
    vae.eval()
    net.eval()

    z_t = torch.randn(1, input_dim)
    for t in reversed(range(timesteps)):
        z_t = scheduler.reverse_diffusion_step(z_t, t, net(z_t))

    # decode
    x_0 = vae.decoder(z_t)
    return x_0
