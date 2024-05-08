---
jupytext:
    cell_metadata_filter: -all
    formats: md:myst
    text_representation:
        extension: .md
        format_name: myst
        format_version: 0.13
        jupytext_version: 1.11.5
mystnb:
    number_source_lines: true
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Stage 5.3. Experiment Tracking And Versioning

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)
![Tag](https://img.shields.io/badge/Level-Beginner-green)

```{contents}
:local:
```

During the iterative process of model development and training, keeping track of
the numerous experiments run, their parameters, and outcomes is a crucial step.
As models are trained, tested, tweaked, and retrained, it becomes increasingly
complex to manage and compare these various experiments.

This is where experiment tracking comes into play. By tracking each experiment,
data scientists can easily compare the results of different models,
configurations, hyperparameters, and even completely different approaches. Also
it enables error analysis, debugging, and model improvement.

## Experiment Tracking

What do we track then? Can we track everything? Well if you can, that's great,
but it's often better and more concise to track things that matter the most for
monitoring and **_debugging_**. Here are some common metrics to track in
experiment tracking:

```{list-table} Some Common Metrics to Track in Experiment Tracking
:header-rows: 1
:name: ml-lifecycle-experiment-tracking-metrics

-   -   Aspect
    -   Description
-   -   Model Architecture
    -   The type of model used, including the number of layers, activation
        functions, and other architectural details.
-   -   Hyperparameters
    -   The settings and hyperparameters used for each model.
-   -   Evaluation Metrics
    -   How each model performed according to the selected evaluation metrics.
-   -   Feature Importance
    -   Which features were most influential in the model's predictions.
-   -   System Metrics
    -   Resource usage, training time, and other system-level metrics such as
        what GPU was used, how much memory was used, etc.
-   -   Gradient Norm (Global and Per Layer), Activation Distribution and Norms,
        Weight Distribution etc
    -   These are important indicators of how well the model is training, and
        can be used to diagnose issues like vanishing gradients, exploding
        gradients, etc. And for example if your embedding layer keep exploding,
        you may want to initialize the weights with smaller values.
```

It's also worth noting that several tools can facilitate experiment tracking,
such as MLflow, TensorBoard, and Weights & Biases. By adopting such tools, teams
can create a central repository of experiments that enable collaboration and
reproducibility. It becomes easier to revisit old experiments, share findings
with team members, and ultimately make more informed decisions about which
models and configurations to move forward with.

If you can, tracking as many key indicators and metrics as possible is a good
way to help debug model issues (i.e. why is model diverging at the 100th step)
and memory leak issues (i.e. why is the CUDA memory usage increasing over time).

## Reproducibility

To ensure that your machine learning experiments are reproducible, you should
keep track of the following components:

1. **Code**
2. **Data**
3. **Model config, artifacts and metadata**
4. **Environment**
5. **Seeding**

### Model Versioning, Code Versioning, and Data Versioning

In addition to tracking experiments, it's also important to version the models,
code, and data used in those experiments. This ensures that the results of an
experiment can be reproduced at a later time, even if the code, data, or
environment have changed.

Now it is worth mentioning that tracking model is the key, and since model is a
combination of code and data, it is important to track the code and data as
well[^chip-chapter6]. Tracking code can be easily done via version control
systems like Git, and tracking data can be done via data versioning tools like
DVC.

Below we see some pseudo code on how to track the code, data, and model
artifacts.

#### 1. Code versioning

Use a version control system like **Git** to keep track of your codebase. Git
allows you to track changes in your code over time and manage different
versions. To log the exact commit hash of your codebase when logging your MLflow
run, you can use the following code snippet:

```python
import subprocess

commit_hash = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
)
mlflow.log_param("commit_hash", commit_hash)
```

By logging the commit hash, you can always refer back to the exact version of
the code used for a specific run, ensuring reproducibility.

#### 2. Data versioning

For data versioning, you can use a tool like **DVC (Data Version Control)**. DVC
is designed to handle large data files, models, and metrics, and it integrates
well with Git. DVC helps you track changes in your data files and manage
different versions.

When you start a new MLflow run, log the DVC version or metadata of the input
data used in the experiment. This way, you can always retrieve the exact version
of the data used for a specific run, ensuring reproducibility.

See [Data Management Tutorial](https://dvc.org/doc/start/data-management) for
more insights.

Important points to consider.

-   gitignore will be created automatically in data folder once you dvc add.
-   After successfully pushing the data to remote, how do you "retrieve them"?
-   If you are in the same repository, you can just pull the data from remote.

The idea is to use dvc checkout to switch between different versions of your
data files, as tracked by DVC. When you use dvc checkout, you provide a Git
commit hash or tag. DVC will then update your working directory with the data
files that were tracked at that specific Git commit.

Here are the steps to use dvc checkout with a Git commit hash:

-   Make sure you have the latest version of your repository and DVC remote by
    running git pull and dvc pull.
-   Switch to the desired Git commit by running git checkout `<commit-hash>`.
-   Run dvc checkout to update your data files to the version tracked at the
    specified commit.

Remember that dvc checkout only updates the data files tracked by DVC. To switch
between code versions, you'll still need to use git checkout.

```bash
git checkout <commit_hash>
dvc checkout # in this commit hash
dvc pull
```

#### 3. Model artifacts and metadata

You have already logged the artifacts (model, vectorizer, config, log files)
using `mlflow.log_artifact()`. You can also log additional metadata related to
the artifacts as you have done with additional_metadata. This should be
sufficient for keeping track of the artifacts associated with each run.

#### Recovering a run

1. Check the commit hashes for the code and data used in the run.
2. Checkout the code and data versions using the commit hashes.

```bash
git checkout <commit_hash>
pip install -r requirements.txt
python main.py train
# once done
git checkout main
```

By combining code versioning with Git, data versioning with DVC, and logging
artifacts and metadata with MLflow, you can ensure that your machine learning
experiments are reproducible. This means that you can always go back and
reproduce the results of a specific experiment, even if the code, data, or
environment have changed but is it always the case? We see that in the next
section.

### Seeding

I won't go into too much on this, but beyond versioning, one must ensure
aggresive seeding in their code base, especially in non-deterministic operations
like training deep learning models. This is however not so simple, even with
aggresive seeding, the same code might produce slightly different results if
trained on a different hardware. Furthermore, a common mistake in the resumption
of training is not saving the rng states, and since dataloaders in frameworks
like pytorch will shuffle (if set to true) on each epoch, it may come as a shock
that resuming training will produce different results.

Just have a look at the below seeding functions I use for single node single GPU
training:

```python
def configure_deterministic_mode() -> None:
    # fmt: off
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark        = False
    torch.backends.cudnn.deterministic    = True
    torch.backends.cudnn.enabled          = False

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # fmt: on
    warnings.warn(
        "Deterministic mode is activated. This will negatively impact performance and may cause increase in CUDA memory footprint.",
        category=UserWarning,
        stacklevel=2,
    )


def seed_all(
    seed: int = 1992,
    seed_torch: bool = True,
    set_torch_deterministic: bool = True,
) -> int:
    # fmt: off
    os.environ["PYTHONHASHSEED"] = str(seed)       # set PYTHONHASHSEED env var at fixed value
    np.random.default_rng(seed)                    # numpy pseudo-random generator
    random.seed(seed)                              # python's built-in pseudo-random generator

    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)           # pytorch (both CPU and CUDA)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

        if set_torch_deterministic:
            configure_deterministic_mode()
    # fmt: on
    return seed
```

And seeding in distributed training is much more complex, so to fully ensure a
reproducible training, one must not only ensure good versioning practices, but
also ensure the seeding mechanism is in place.

## References and Further Readings

-   Huyen, Chip. "Chapter 6. Model Development and Offline Evaluation." In
    Designing Machine Learning Systems: An Iterative Process for
    Production-Ready Applications, O'Reilly Media, Inc., 2022.

[^chip-chapter6]:
    Huyen, Chip. "Chapter 6. Model Development and Offline Evaluation." In
    Designing Machine Learning Systems: An Iterative Process for
    Production-Ready Applications, O'Reilly Media, Inc., 2022.
