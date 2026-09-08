"""MNIST training."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import tqdm
import tyro


@dataclass
class Config:
    batch_size: int = 64
    learning_rate: float = 1e-2
    epochs: int = 15


class MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.dout1 = nn.Dropout(0.25)
        self.dout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4608, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dout2(x)
        x = self.fc2(x)
        return x


def get_data_loaders(config: Config):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    data_dir = Path(__file__).parents[4] / "data"

    dataset1 = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_kwargs = {"batch_size": config.batch_size, "shuffle": True}
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)

    test_kwargs = {"batch_size": config.batch_size}
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader


def train_step(
    model,
    loss_fn: Callable,
    optimizer,
    data: torch.Tensor,
    target: torch.Tensor,
    step_id: int,
    writer,
):
    optimizer.zero_grad()
    output = model(data)  # output is logits.
    loss = loss_fn(output, target)  # target can be an index or one-hot vector.
    loss.backward()
    optimizer.step()
    writer.add_scalar("Loss/train", loss, global_step=step_id)


def train_epoch(model, loss_fn: Callable, optimizer, train_loader, epoch: int, writer):
    model.train()
    for idx, (data, target) in enumerate(tqdm.tqdm(train_loader, desc="Train"), 1):
        step_id = epoch * len(train_loader) + idx
        train_step(model, loss_fn, optimizer, data, target, step_id, writer)


def eval(model, test_loader, epoch: int, writer):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in tqdm.tqdm(test_loader, desc="Eval"):
        pred = model(data)
        test_loss += F.cross_entropy(pred, target, reduction="sum").item()
        pred = torch.argmax(pred, dim=1, keepdims=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    num_items = len(test_loader.dataset)
    test_loss /= num_items
    accuracy = correct / num_items

    writer.add_scalar("Loss/eval", test_loss, epoch + 1)
    writer.add_scalar("Accuracy/eval", accuracy, epoch + 1)


def main(config: Config) -> None:
    model = MNIST()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    train_loader, test_loader = get_data_loaders(config)
    writer = SummaryWriter()

    for epoch in range(config.epochs):
        train_epoch(model, loss_fn, optimizer, train_loader, epoch, writer)
        eval(model, test_loader, epoch, writer)

    writer.close()


if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
