# Two Digits Adder

-   [Two Digits Adder](#two-digits-adder) -
    [Run The Pipeline](#run-the-pipeline)
    -   [Experiments](#experiments)
        -   [Run 1. CPU Bound 3 Epochs (Debug)](#run-1-cpu-bound-3-epochs-debug)
        -   [Run 2. CPU Bound 20 Epochs](#run-2-cpu-bound-20-epochs)
        -   [Run 3. CPU Bound 20 Epochs with Automatic Mixed Precision](#run-3-cpu-bound-20-epochs-with-automatic-mixed-precision)
        -   [Run 4. CPU Bound 20 Epochs with Automatic Mixed Precision and Gradient Scaler](#run-4-cpu-bound-20-epochs-with-automatic-mixed-precision-and-gradient-scaler)
        -   [Run 5. GPU Bound 30 Epochs with Automatic Mixed Precision and Gradient Scaler](#run-5-gpu-bound-30-epochs-with-automatic-mixed-precision-and-gradient-scaler)
        -   [Run X: Gradient Accumulation](#run-x-gradient-accumulation)

### Run The Pipeline

```bash
python omnivault/transformer/projects/adder/main.py \
    omnivault/transformer/projects/adder/config.yaml \
    data.train_loader.batch_size=256 \
    data.valid_loader.batch_size=256

# if weight decay is 0, then it is as good as not applying custom weight decay to diff param groups:
python omnivault/transformer/projects/adder/main.py omnivault/transformer/projects/adder/config.yaml data.train_loader.batch_size=256 data.valid_loader.batch_size=256 trainer.apply_weight_decay_to_different_param_groups=True optimizer.weight_decay=1e-2
```

To test the "generalization", we can ask some questions that are not in the
training set:

```bash
97+98=195
96+96=192
95+95=190
```

but we do not really need to do this since we split into `train-valid-test`
already, and in a sense, the `valid` and `test` sets are "unseen" by the model,
acting as a rough holdout.

> Important, we must use greedy generation and not top-k or top-p (nuclues)
> sampling here because we really just want the model to output the exact
> answer, and not some other answer that is close to the correct answer in the
> distribution of the model's vocabulary.

## Experiments
