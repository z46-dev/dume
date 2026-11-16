# MNIST example

This example trains a small dense classifier using the already downloaded
20x20 digit dataset at `data/training/Mnist_0_to_9_20x20_5000.csv`. The CSV has
400 normalized pixel columns and a final `label` column. It makes a reproducible
shuffled train/test split, trains on the training portion, then reports overall
and per-digit accuracy on the held-out portion.

Run from the repository root:

```sh
go run ./examples/mnist
```

Defaults are 20 epochs, batches of 32, an 80/20 split, and a learning rate of
`0.1`. The classifier uses a ReLU hidden layer and a Softmax output trained with
categorical cross-entropy. Use `-csv PATH`, `-epochs N`, `-batch-size N`,
`-train-percent N`, and `-learning-rate RATE` to adjust the run.
