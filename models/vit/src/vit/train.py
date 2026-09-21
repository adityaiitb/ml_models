"""ViT training on CIFAR-10"""

import os


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tqdm
import tyro


from vit.config import Config
from vit.model import ViT
from vit.lr_schedular import get_schedular
from vit.dataloader import get_dataloaders


def restore_ckpt(config: Config, model, optimizer, schedular) -> tuple[int, int]:
    """Read a ckpt and restore the model, optimizer, and schedular. Return the epoch and global_step."""

    ckpt_path = os.path.join(config.checkpoint.dir, "checkpoint.pt")

    epoch = 0
    global_step = -1

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        schedular.load_state_dict(ckpt["schedular_state_dict"])
        print(
            f"Found {ckpt_path} at epoch {epoch} and global step {global_step}. Restoring."
        )
    else:
        print(f"No checkpoint found. Starting from scratch.")
    return (epoch, global_step)


def save_ckpt(config: Config, model, optimizer, schedular, epoch, global_step):
    ckpt_path_tmp = os.path.join(config.checkpoint.dir, "checkpoint.pt.tmp")
    ckpt_path = os.path.join(config.checkpoint.dir, "checkpoint.pt")

    ckpt_data = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "schedular_state_dict": schedular.state_dict(),
    }

    torch.save(ckpt_data, ckpt_path_tmp)
    os.replace(ckpt_path_tmp, ckpt_path)
    print(f"Saved {ckpt_path} at epoch {epoch} global step {global_step}.")


def train_step(model, loss_fn, optimizer, schedular, images, labels, step_id, writer):
    model.train()
    # Log current LR
    lr = schedular.get_last_lr()[0]

    optimizer.zero_grad()
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    loss.backward()

    # Clip gradients to prevent explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    schedular.step()

    writer.add_scalar("LR/train", lr, step_id)
    writer.add_scalar("Loss/train", loss, step_id)


def eval(model, test_loader, global_step, writer):
    model.eval()
    num_items = len(test_loader.dataset)
    loss = 0
    correct = 0

    for images, labels in tqdm.tqdm(test_loader, desc="Eval"):
        outputs = model(images)
        loss += F.cross_entropy(outputs, labels, reduction="sum").item()
        pred = torch.argmax(outputs, dim=1, keepdims=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

    loss /= num_items
    accuracy = correct / num_items
    writer.add_scalar("Loss/eval", loss, global_step)
    writer.add_scalar("Accuracy/eval", accuracy, global_step)


def main(config: Config):
    model = ViT(config.model)
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter(config.train.tensorboard_dir)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr_schedule.lr, weight_decay=0.1
    )
    train_loader, test_loader = get_dataloaders(config)
    schedular = get_schedular(config, optimizer, len(train_loader))
    steps_per_epoch = len(train_loader)

    # Restore ckpt if one exists.
    os.makedirs(config.checkpoint.dir, exist_ok=True)
    ckpt_epoch, ckpt_global_step = restore_ckpt(config, model, optimizer, schedular)

    for epoch in range(ckpt_epoch, config.train.epochs):
        for idx, (images, labels) in enumerate(tqdm.tqdm(train_loader, desc="Train")):
            global_step = epoch * steps_per_epoch + idx

            if global_step <= ckpt_global_step:
                continue

            train_step(
                model,
                loss_fn,
                optimizer,
                schedular,
                images,
                labels,
                global_step,
                writer,
            )

            # Eval
            if global_step % config.train.eval_steps == 0:
                eval(model, test_loader, global_step, writer)

            # Save checkpoint.
            if (global_step) and (global_step % config.checkpoint.steps == 0):
                save_ckpt(config, model, optimizer, schedular, epoch, global_step)

    writer.close()


if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
