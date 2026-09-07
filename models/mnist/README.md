```sh
cd src/mnist
uv run python mnist.py
```

The accuracy after 15 epochs is about 98.8%.
Use Tensorboard `uv run tensorboard --logdir <runs/...>` to visualize loss and accuracy.
