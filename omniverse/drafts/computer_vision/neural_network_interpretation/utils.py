import logging
import os
import random
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    https://github.com/FrancescoSaverioZuppichini/A-journey-into-Convolutional-Neural-Network-visualization-
    """

    def __init__(self, mean, std):
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def seed_all(seed: int = 1992) -> None:
    """Seed all random number generators."""
    print(f"Using Seed Number {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    # set fixed value for python built-in pseudo-random generator
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def seed_worker(_worker_id) -> None:
    """Seed a worker with the given ID."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def init_logger(log_file: str = "info.log") -> logging.Logger:
    """Initialize logger and save to file.

    Consider having more log_file paths to save, eg: debug.log, error.log, etc.

    Args:
        log_file (str, optional): [description]. Defaults to Path(LOGS_DIR, "info.log").

    Returns:
        logging.Logger: [description]
    """
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    stream_handler = StreamHandler()
    stream_handler.setFormatter(Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    file_handler = FileHandler(filename=log_file)
    file_handler.setFormatter(Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def tensor2img(tensor, ax=plt):
    tensor = tensor.squeeze()
    if len(tensor.shape) > 2:
        tensor = tensor.permute(1, 2, 0)
    img = tensor.detach().cpu().numpy()
    return img


def subplot(
    images,
    parse=lambda x: x,
    rows_titles=None,
    cols_titles=None,
    title="",
    *args,
    **kwargs,
):
    fig, ax = plt.subplots(*args, **kwargs)
    fig.suptitle(title)
    i = 0
    try:
        for row in ax:
            if rows_titles is not None:
                row.set_title(rows_titles[i])
            try:
                for j, col in enumerate(row):
                    if cols_titles is not None:
                        col.set_title(cols_titles[j])
                    col.imshow(parse(images[i]))
                    col.axis("off")
                    col.set_aspect("equal")
                    i += 1
            except TypeError:
                row.imshow(parse(images[i]))
                row.axis("off")
                row.set_aspect("equal")
                i += 1
            except IndexError:
                break

    except:
        ax.imshow(parse(images[i]))
        ax.axis("off")
        ax.set_aspect("equal")

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        plt.show()


def module2traced(module, inputs):
    handles, modules = [], []

    def trace(module, inputs, outputs):
        modules.append(module)

    def traverse(module):
        for m in module.children():
            traverse(m)  # recursion is love
        is_leaf = len(list(module.children())) == 0
        if is_leaf:
            handles.append(module.register_forward_hook(trace))

    traverse(module)

    _ = module(inputs)

    [h.remove() for h in handles]

    return modules


def run_vis_plot(vis, x, layer, ncols=1, nrows=1):
    images, info = vis(x, layer)
    images = images[: nrows * ncols]
    print(images[0].shape)
    subplot(images, tensor2img, title=str(layer), ncols=ncols, nrows=nrows)


def run_vis_plot_across_models(
    modules,
    input,
    layer_id,
    Vis,
    title,
    device,
    inputs=None,
    nrows=3,
    ncols=2,
    row_wise=True,
    parse=tensor2img,
    annotations=None,
    idx2label=None,
    rows_name=None,
    *args,
    **kwargs,
):
    pad = 0  # in points
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    fig.suptitle(title)

    for i, row in enumerate(ax):
        try:
            module = next(modules)
            module.eval()
            module = module.to(device)
            layer = None
            if layer_id is not None:
                layer = module2traced(module, input)[layer_id]
            vis = Vis(module, device)
            info = {}
            if inputs is None:
                images, info = vis(input.clone(), layer, *args, **kwargs)
            row_title = module.__class__.__name__
            del module
            torch.cuda.empty_cache()
            if rows_name is not None:
                row_title = rows_name[i]
            row[0].set_title(row_title)
            if annotations is not None:
                row[0].annotate(
                    annotations[i],
                    xy=(0, 0.5),
                    xytext=(-row[0].yaxis.labelpad - pad, 0),
                    xycoords=row[0].yaxis.label,
                    textcoords="offset points",
                    size="medium",
                    ha="right",
                    va="center",
                    rotation=90,
                )
            for j, col in enumerate(row):
                if inputs is None:
                    image = images[j]
                else:
                    image, info = vis(inputs[j], layer, *args, **kwargs)
                if "prediction" in info:
                    col.set_title(idx2label[int(info["prediction"])])
                col.imshow(parse(image))
                col.axis("off")
                col.set_aspect("equal")
        except StopIteration:
            break
        except:
            row.set_title(row_title)
            row.imshow(parse(images[0]))
            row.axis("off")
            row.set_aspect("equal")
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
