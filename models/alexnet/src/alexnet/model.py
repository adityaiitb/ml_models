"""AlexNet training."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import tqdm
import tyro

# https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py


@dataclass
class Config:
    batch_size: int = 128
    learning_rate: float = 1e-2
    epochs: int = 50


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 2)

        # Layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 2)

        # Layer 3
        x = self.conv3(x)
        x = F.relu(x)

        # Layer 4
        x = self.conv4(x)
        x = F.relu(x)

        # Layer 5
        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 2)

        # Flatten
        x = torch.flatten(x, 1)

        # Linear
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x


class AlexNetCIFAR10(nn.Module):
    def __init__(self, num_classes=10, dropout: float = 0.5):
        super().__init__()
        self.layer1 = nn.Sequential(
            # (3, 32, 32) -> (64, 32, 32) -> (64, 16, 16)
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.layer2 = nn.Sequential(
            # (64, 16, 16) -> (192, 16, 16) -> (192, 8, 8)
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.layer3 = nn.Sequential(
            # (192, 8, 8) -> (384, 8, 8)
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            # (384, 8, 8) -> (256, 8, 8)
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer5 = nn.Sequential(
            # (256, 8, 8) -> (256, 8, 8) -> (256, 4, 4)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.mlp1 = nn.Sequential(
            # (256, 4, 4) -> (4096) -> (1024)
            nn.Dropout(dropout),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            # (1024) -> (1024)
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )
        self.mlp3 = nn.Sequential(
            # (1024) -> (num_classes)
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, 1)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        return x


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
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    writer.add_scalar("Loss/train", loss, step_id)


def train_epoch(model, loss_fn: Callable, optimizer, train_loader, epoch: int, writer):
    model.train()
    for idx, (data, target) in enumerate(tqdm.tqdm(train_loader, desc="Train"), 1):
        step_id = epoch * len(train_loader) + idx
        train_step(model, loss_fn, optimizer, data, target, step_id, writer)


def eval(model, test_loader, epoch: int, writer):
    model.eval()
    loss = 0
    correct = 0
    for data, target in tqdm.tqdm(test_loader, desc="Eval"):
        output = model(data)
        loss += F.cross_entropy(output, target, reduction="sum").item()
        pred = torch.argmax(output, dim=1, keepdims=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    num_items = len(test_loader.dataset)
    loss /= num_items
    accuracy = correct / num_items
    writer.add_scalar("Loss/eval", loss, epoch + 1)
    writer.add_scalar("Accuracy/eval", accuracy, epoch + 1)


def get_data_loaders(config: Config):
    # Transform for CIFAr10
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
            ),
        ]
    )

    data_dir = Path(__file__).parents[4] / "data"

    dataset1 = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=transform
    )
    dataset2 = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=transform
    )

    train_kwargs = {"batch_size": config.batch_size, "shuffle": True}
    train_loader = DataLoader(dataset1, **train_kwargs)

    test_kwargs = {"batch_size": config.batch_size}
    test_loader = DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader


def main(config: Config):
    # model = AlexNet()
    model = AlexNetCIFAR10()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    writer = SummaryWriter()
    train_loader, test_loader = get_data_loaders(config)

    for epoch in range(config.epochs):
        train_epoch(model, loss_fn, optimizer, train_loader, epoch, writer)
        eval(model, test_loader, epoch, writer)


if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
