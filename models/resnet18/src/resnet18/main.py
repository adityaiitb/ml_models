"""ResNet18 training on CIFAR-10."""

from dataclasses import dataclass
import os
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


@dataclass
class Config:
    batch_size: int = 32
    learning_rate: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 1e-4
    epochs: int = 50
    ckpt_dir: str = "ckpt/"
    tensorboard_dir: str = "runs/resnet18"


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.shortcut = nn.Identity()

        # Fix the shortcut connection when either the
        # number of i/p and o/p channels are different, or
        # the stride changes (e.g. downsampleing), or
        # both.
        if (stride != 1) or (in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += identity
        x = F.relu(x)
        return x


class ResNet18CIFAR10(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # (3, 32, 32) -> (64, 32, 32)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        # (64, 32, 32) -> (64, 32, 32)
        self.conv2_x = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))

        # (64, 32, 32) -> (128, 32, 32)
        self.conv3_x = nn.Sequential(BasicBlock(64, 128), BasicBlock(128, 128))

        # (128, 32, 32) -> (256, 16, 16)
        self.conv4_x = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256, 1))

        # (256, 16, 16) -> (512, 8, 8)
        self.conv5_x = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512, 1))

        # (512, 8, 8) -> (512, 1, 1)
        self.avgpool = nn.AvgPool2d(8)

        # (512) -> num_classes
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


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


def train_step(model, loss_fn, optimizer, data, target, step_id, writer):
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    writer.add_scalar("Loss/train", loss, step_id)


def train_epoch(model, loss_fn, optimizer, train_loader, epoch, writer):
    model.train()
    for idx, (data, target) in enumerate(tqdm.tqdm(train_loader, desc="Train")):
        step_id = epoch * len(train_loader) + idx
        train_step(model, loss_fn, optimizer, data, target, step_id, writer)


def eval(model, test_loader, epoch, writer):
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


def main(config: Config):
    model = ResNet18CIFAR10()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    writer = SummaryWriter(config.tensorboard_dir)
    train_loader, test_loader = get_data_loaders(config)
    start_epoch = 0

    os.makedirs(config.ckpt_dir, exist_ok=True)
    ckpt_path_tmp = os.path.join(config.ckpt_dir, "checkpoint.pt.tmp")
    ckpt_path = os.path.join(config.ckpt_dir, "checkpoint.pt")

    # Checkpoint restore
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Found {ckpt_path} at epoch {start_epoch - 1}. Restoring.")
    else:
        print(f"No checkpoint found. Starting from scratch.")

    for epoch in range(start_epoch, config.epochs):
        train_epoch(model, loss_fn, optimizer, train_loader, epoch, writer)
        eval(model, test_loader, epoch, writer)

        # Save checkpoint.
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        torch.save(ckpt_data, ckpt_path_tmp)
        os.replace(ckpt_path_tmp, ckpt_path)
        print(f"Saved {ckpt_path} at epoch {epoch}.")

    writer.close()


if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
