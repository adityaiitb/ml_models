from dataclasses import dataclass


@dataclass
class DatasetConfig:
    path: str
    name: str = "cifar-10"
    # 0 means the entire dataset. Positive number means a subset (for testing).
    samples: int = 0


@dataclass
class DataLoaderConfig:
    shuffle: bool = True
    drop_last: bool = True


@dataclass
class ModelConfig:
    image_shape: tuple[int, int, int] = (3, 32, 32)
    patch_shape: tuple[int, int] = (4, 4)
    emb_dim: int = 256  # 768 used in ViT-base
    heads: int = 4  # 12 used in ViT-base
    head_dim: int = 64
    layers: int = 3  # 12 used in ViT-base
    ffn_hidden_dim: int = 1024  # 3072 ViT-base
    dropout: float = 0.1
    qkv_bias: bool = False
    num_classes: int = 10


@dataclass
class LRScheduleConfig:
    lr: float = 1e-4
    warmup_fraction: float = 0.1


@dataclass
class CheckpointConfig:
    dir: str = "ckpt/"
    steps: int = 50


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 30
    eval_steps: int = 200
    tensorboard_dir: str = "runs/vit"


@dataclass
class Config:
    model: ModelConfig
    dataset: DatasetConfig
    dataloader: DataLoaderConfig
    checkpoint: CheckpointConfig
    train: TrainConfig
    lr_schedule: LRScheduleConfig
