from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from vit.config import Config


def get_dataloaders(config: Config):
    # Transform for CIFAR-10
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
            ),
        ]
    )

    name = config.dataset.name
    path = config.dataset.path
    assert name == "cifar-10"

    dataset1 = datasets.CIFAR10(
        path, train=True, download=True, transform=train_transform
    )
    dataset2 = datasets.CIFAR10(
        path, train=False, download=True, transform=test_transform
    )

    samples = config.dataset.samples
    if samples > 0:
        dataset1 = Subset(dataset1, list(range(samples)))
        dataset2 = Subset(dataset2, list(range(samples)))

    train_kwargs = {
        "batch_size": config.train.batch_size,
        "shuffle": config.dataloader.shuffle,
        "drop_last": config.dataloader.drop_last,
    }
    train_loader = DataLoader(dataset1, **train_kwargs)

    test_kwargs = {"batch_size": config.train.batch_size}
    test_loader = DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader
