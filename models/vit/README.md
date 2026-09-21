Train tiny ViT on CIFAR-10 for 30 epochs using AdamW, warmup and cosine decay LR schedule. The accuracy is about 71% (see Tensorboard at runs/vit).

```sh
uv run python train.py --dataset.path ../../../../data/
```
