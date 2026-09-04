"""MNIST training."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        self.fc1 = nn.Linear(4608, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def train_step(
    model, loss_fn: Callable, optimizer, data: torch.Tensor, target: torch.Tensor
) -> float:
    optimizer.zero_grad()
    output = model(data)  # output is logits.
    loss = loss_fn(output, target)  # target can be an index or one-hot vector.
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(model, loss_fn: Callable, optimizer, train_loader) -> list[float]:
    model.train()
    loss_array = []
    for (data, target) in tqdm.tqdm(train_loader, desc="Training steps"):
        loss = train_step(model, loss_fn, optimizer, data, target)
        loss_array.append(loss.item())
    return loss_array


def eval(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for (data, target) in tqdm.tqdm(test_loader, desc="Eval"):
        pred = model(data)
        test_loss += F.cross_entropy(pred, target, reduction="sum").item()
        pred = torch.argmax(pred, dim=1, keepdims=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    num_items = len(test_loader.dataset)
    test_loss /= num_items
    accuracy = correct / num_items

    return (test_loss, accuracy)


def main(config: Config) -> None:
    model = MNIST()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    )

    dataset1 = datasets.MNIST(
        "../../data", train=True, download=True, transform=transform
    )
    dataset2 = datasets.MNIST(
        "../../data", train=False, download=True, transform=transform
    )

    train_kwargs = {"batch_size": config.batch_size, "shuffle": True}
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)

    test_kwargs = {"batch_size": config.batch_size}
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    train_loss = []
    eval_data = []
    for _ in tqdm.tqdm(range(config.epochs), desc="Training epochs"):
        loss = train_epoch(model, loss_fn, optimizer, train_loader)
        train_loss.extend(loss)
        loss, accuracy = eval(model, test_loader)
        eval_data.append((loss, accuracy))

    with open("train.dat", "w") as f:
        for loss in train_loss:
            f.write(f"{round(loss, 5)}\n")

    with open("test.dat", "w") as f:
        for loss, acc in eval_data:
            f.write(f"{round(loss, 5)} {round(acc, 5)}\n")


if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
