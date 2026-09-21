from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


from vit.config import Config


def get_schedular(config: Config, optimizer, steps_per_epoch):
    epochs = config.train.epochs
    fraction = config.lr_schedule.warmup_fraction
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * fraction)

    warmup = LinearLR(
        optimizer, start_factor=0.01, end_factor=1, total_iters=warmup_steps
    )

    cosine = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
    )

    return SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )
