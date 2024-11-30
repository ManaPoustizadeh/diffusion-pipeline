import torch
import torchvision
from torch.utils.data import DataLoader
from torch.nn import MSELoss

from simple_unet.simple_unet import SimpleUNet


def get_data_loader(batch_size: int = 120) -> DataLoader:
    dataset = torchvision.datasets.MNIST(
        root="datasets/mnist/",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader


def corrupt(x, amount):
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)
    return x * (1 - amount) + amount * noise


def train(net: SimpleUNet, train_dataloader: DataLoader, epochs: int = 3):
    losses = []
    loss_f = MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(epochs):
        for x, y in train_dataloader:
            noise_amount = torch.rand(x.shape[0])

            x = corrupt(x, noise_amount)

            prediction = net(x)

            loss = loss_f(x, prediction)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss)

        loss_avg = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
        print(f"Finished epoch {epoch}. Average loss for this epoch: {loss_avg:05f}")


if __name__ == "__main__":
    unet = SimpleUNet()
    train_dataloader = get_data_loader()
    train(unet, train_dataloader, 6)
    # my avg loss with these params:
    # Finished epoch 0. Average loss for this epoch: 0.023570
    # Finished epoch 1. Average loss for this epoch: 0.001357
    # Finished epoch 2. Average loss for this epoch: 0.000986
    # Finished epoch 3. Average loss for this epoch: 0.000815
    # Finished epoch 4. Average loss for this epoch: 0.000656
    # Finished epoch 5. Average loss for this epoch: 0.000529
