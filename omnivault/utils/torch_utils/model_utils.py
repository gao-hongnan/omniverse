from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple, TypedDict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
from torch import nn

__all__ = [
    "total_trainable_parameters",
    "total_parameters",
    "compare_models",
    "compare_models_and_report_differences",
    "get_named_modules",
    "gather_weight_stats",
    "prepare_stats_dataframe",
    "plot_distribution_stats",
    "get_named_parameters",
    "Freezer",
    "freeze_layers",
    "check_optimizer_coverage",
    "get_model_layer_info",
]


def total_trainable_parameters(module: nn.Module) -> int:
    """Returns the number of trainable parameters in the model."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def total_parameters(module: nn.Module) -> int:
    """Returns the total number of parameters in the model, including non-trainable."""
    return sum(p.numel() for p in module.parameters())


def get_named_modules(module: nn.Module, **kwargs: Any) -> List[Dict[str, str]]:
    """Obtain a list of named modules in the model/module.

    Return an iterator over all modules in the network, yielding both the name
    of the module as well as the module itself.

    Parameters
    ----------
    module : nn.Module
        The model to extract named modules from.
    **kwargs : Any
        Additional keyword arguments to pass to the `named_modules` method.

    Returns
    -------
    List[Dict[str, str]]
        A list of dictionaries containing the name and type of each module in the model.

    Examples
    --------
    >>> from torchvision.models import resnet18, ResNet18_Weights
    >>> backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    >>> named_modules = get_named_modules(backbone)

    ```python
    some_module = nn.ModuleDict(
        {"backbone": nn.Sequential(
            nn.Embedding(2, 2),
        ),
        "pooler":
            nn.Sequential(
                nn.Linear(1, 2),
            ),
        "head": nn.Sequential(
            nn.Conv2d(1, 3, 1),
        )
    })
    ```

    The model definition is:

    ```python
    ModuleDict(
    (backbone): Sequential(
        (0): Embedding(2, 2)
    )
    (pooler): Sequential(
        (0): Linear(in_features=1, out_features=2, bias=True)
    )
    (head): Sequential(
        (0): Conv2d(1, 3, kernel_size=(1, 1), stride=(1, 1))
    )
    )
    ```

    Then if we call `get_named_modules(some_module)`, we will get the following output:
    ```python
    [
        {
            "": "ModuleDict(backbone): Sequential((0): Embedding(2, 2))pooler): Sequential((0): Linear(in_features=1, out_features=2, bias=True))head): Sequential((0): Conv2d(1, 3, kernel_size=(1, 1), stride=(1, 1)))\n)"
        },
        {"backbone": "Sequential(0): Embedding(2, 2)\n)"},
        {"backbone.0": "Embedding(2, 2)"},
        {"pooler": "Sequential(0): Linear(in_features=1, out_features=2, bias=True)\n)"},
        {"pooler.0": "Linear(in_features=1, out_features=2, bias=True)"},
        {"head": "Sequential(0): Conv2d(1, 3, kernel_size=(1, 1), stride=(1, 1))\n)"},
        {"head.0": "Conv2d(1, 3, kernel_size=(1, 1), stride=(1, 1))"},
    ]
    ```
    """
    named_modules = []
    for _module in module.named_modules(**kwargs):
        module_name, module_type = _module
        named_modules.append({str(module_name): str(module_type)})
    return named_modules


def get_named_parameters(module: nn.Module, **kwargs: Any) -> List[Dict[str, nn.Parameter]]:
    """Obtain a list of named parameters in the model/module.

    Return an iterator over all parameters in the network, yielding both the name
    of the parameter as well as the parameter itself.

    Parameters
    ----------
    module : nn.Module
        The model to extract named parameters from.
    **kwargs : Any
        Additional keyword arguments to pass to the `named_parameters` method.

    Returns
    -------
    List[Dict[str, nn.Parameter]]
        A list of dictionaries containing the name and parameter of each parameter in the model.

    Examples
    --------
    >>> from torchvision.models import resnet18, ResNet18_Weights
    >>> backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    >>> named_parameters = get_named_parameters(backbone)

    ```python
    some_module = nn.ModuleDict(
        {"backbone": nn.Sequential(
            nn.Embedding(2, 2),
        ),
        "pooler":
            nn.Sequential(
                nn.Linear(1, 2),
            ),
        "head": nn.Sequential(
            nn.Conv2d(1, 3, 1),
        )
    })
    ```

    The model definition is:

    ```python
    ModuleDict(
    (backbone): Sequential(
        (0): Embedding(2, 2)
    )
    (pooler): Sequential(
        (0): Linear(in_features=1, out_features=2, bias=True)
    )
    (head): Sequential(
        (0): Conv2d(1, 3, kernel_size=(1, 1), stride=(1, 1))
    )
    )
    ```

    Then if we call `get_named_parameters(some_module)`, we will get the following output:

    ```text
    [
        {
            'backbone.0.weight': Parameter containing:
            tensor([
                [ 0.4833, -0.9517],
                [ 0.7517, -0.3315]
            ], requires_grad=True)
        },
        {
            'pooler.0.weight': Parameter containing:
            tensor([
                [ 0.8330],
                [-0.0932]
            ], requires_grad=True)
        },
        {
            'pooler.0.bias': Parameter containing:
            tensor([
                -0.5987,
                0.2774
            ], requires_grad=True)
        },
        {
            'head.0.weight': Parameter containing:
            tensor([
                [[[-0.3349]]],
                [[[-0.4394]]],
                [[[ 0.2738]]]
            ], requires_grad=True)
        },
        {
            'head.0.bias': Parameter containing:
            tensor([
                0.5669,
                0.7181,
                -0.0820
            ], requires_grad=True)
        }
    ]
    ```
    """
    named_parameters = []
    for _parameter in module.named_parameters(**kwargs):
        parameter_name, parameter = _parameter
        named_parameters.append({str(parameter_name): parameter})
    return named_parameters


def gather_weight_stats(module: nn.Module, **kwargs: Any) -> Dict[str, Dict[str, float]]:
    """Return the mean and standard deviation of weights and biases in the model. Sanity
    check to ensure that the weights and biases are initialized correctly.

    Parameters
    ----------
    module : nn.Module
        The model to extract weight statistics from.
    **kwargs : Any
        Additional keyword arguments to pass to the `named_modules` method.

    Returns
    -------
    Dict[str, Dict[str, float]]
        A dictionary containing the mean and standard deviation of weights and biases in the model.

    Examples
    --------
    >>> from torchvision.models import resnet18, ResNet18_Weights
    >>> backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    >>> stats = gather_weight_stats(backbone)
    >>> stats_df = prepare_stats_dataframe(stats)
    >>> plot_distribution_stats(stats_df)
    >>> plot_distribution_stats(stats_df, layer_type="Conv2d")
    """
    stats = {}
    for _module in module.named_modules(**kwargs):
        module_name, module_type = _module
        assert isinstance(module_type, nn.Module)
        if (
            hasattr(module_type, "weight")
            and isinstance(module_type.weight, torch.nn.Parameter)
            and module_type.weight is not None
        ):
            weight = module_type.weight.data
            weight_key = f"{module_name}+{str(module_type.__class__.__name__)}_weight"
            stats[weight_key] = {"w_mean": weight.mean().item(), "w_std": weight.std().item()}

        if (
            hasattr(module_type, "bias")
            and isinstance(module_type.bias, torch.nn.Parameter)
            and module_type.bias is not None
        ):
            bias = module_type.bias.data
            bias_key = f"{module_name}+{str(module_type.__class__.__name__)}_bias"
            stats[bias_key] = {"b_mean": bias.mean().item(), "b_std": bias.std().item()}
    return stats


def prepare_stats_dataframe(stats_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Use in conjunction with the `gather_weight_stats` function to prepare a DataFrame
    for visualization of weight and bias statistics."""
    data = []
    for key, values in stats_dict.items():
        layer_name, layer_type = key.rsplit("+", 1)
        layer_name = "+".join(layer_name.split("+")[:-1])
        layer_type = layer_type.replace("_weight", "").replace("_bias", "")

        data.append(
            {
                "layer_name": layer_name,
                "layer_type": layer_type,
                "parameter_type": "Weight" if "weight" in key else "Bias",
                "mean": values["w_mean"] if "weight" in key else values["b_mean"],
                "std": values["w_std"] if "weight" in key else values["b_std"],
            }
        )
    return pd.DataFrame(data)


def plot_distribution_stats(df: pd.DataFrame, layer_type: str | None = None) -> None:
    """Use in conjunction with the `prepare_stats_dataframe` function to visualize the
    distribution of weight and bias statistics."""
    if layer_type:
        df = df[df["layer_type"] == layer_type]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    sns.histplot(df[df["parameter_type"] == "Weight"], x="mean", kde=True, color="blue", ax=axes[0, 0])
    axes[0, 0].set_title("Distribution of Weights Means")
    sns.histplot(df[df["parameter_type"] == "Weight"], x="std", kde=True, color="blue", ax=axes[0, 1])
    axes[0, 1].set_title("Distribution of Weights Standard Deviations")

    sns.histplot(df[df["parameter_type"] == "Bias"], x="mean", kde=True, color="red", ax=axes[1, 0])
    axes[1, 0].set_title("Distribution of Biases Means")
    sns.histplot(df[df["parameter_type"] == "Bias"], x="std", kde=True, color="red", ax=axes[1, 1])
    axes[1, 1].set_title("Distribution of Biases Standard Deviations")

    plt.tight_layout()
    plt.show()


def compare_models(model_a: nn.Module, model_b: nn.Module) -> bool:
    """
    Compare two PyTorch models to check if they have identical parameters.

    Parameters
    ----------
    model_a : nn.Module
        The first model to compare.
    model_b : nn.Module
        The second model to compare.

    Returns
    -------
    bool
        Returns True if both models have identical parameters, False otherwise.
    """
    return all(
        torch.equal(param_a[1], param_b[1])
        for param_a, param_b in zip(model_a.state_dict().items(), model_b.state_dict().items())
    )


def compare_models_and_report_differences(model_a: nn.Module, model_b: nn.Module) -> Tuple[bool, Any]:
    """
    Compare two PyTorch models to check if they have identical parameters.

    Parameters
    ----------
    model_a : nn.Module
        The first model to compare.
    model_b : nn.Module
        The second model to compare.

    Returns
    -------
    Tuple[bool, Any]
        Returns a tuple with the first element as a boolean indicating if the
        models are identical. The second element is a dictionary containing the
        differences between the models if they are not identical.
    """
    model_a_dict = model_a.state_dict()
    model_b_dict = model_b.state_dict()

    if set(model_a_dict.keys()) != set(model_b_dict.keys()):
        # Early exit if model architectures are different (different sets of parameter keys)
        return False, {"error": "Models have different architectures and cannot be compared."}

    differences = {}
    for name in model_a_dict.keys():  # noqa: SIM118
        param_a = model_a_dict[name]
        param_b = model_b_dict[name]
        if not torch.equal(param_a, param_b):
            differences[name] = {
                "model_a": param_a.detach().cpu().numpy(),
                "model_b": param_b.detach().cpu().numpy(),
            }

    if differences:
        return False, differences
    else:
        return True, None


def _init_weights(module: nn.Module, **kwargs: Any) -> None:
    """Initialize the weights. This is a helper function to initialize the weights
    of a torch model.

    Parameters
    ----------
    module : nn.Module
        The module to initialize the weights. It can be the model itself, or
        a sub-module of the model such as `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`.
    kwargs : Any
        Additional keyword arguments to pass to the initialization method.
        This is like a _universal_ way to accept configuration.

    Examples
    --------
    >>> from torch import nn
    >>> model = nn.Sequential(
    ...     nn.Linear(10, 10),
    ...     nn.ReLU(),
    ...     nn.Embedding(10, 10),
    ...     nn.LayerNorm(10),
    ... )
    >>> model.apply(lambda module: _init_weights(module, init_method="xavier_uniform", gain=10.0))
    """
    # std = self.config.initializer_range
    init_method = kwargs.get("init_method", "normal")

    if isinstance(module, nn.Linear):
        if init_method == "normal":
            module.weight.data.normal_(mean=kwargs.get("mean", 0.0), std=kwargs.get("std", 0.02))
        elif init_method == "xavier_uniform":
            module.weight.data = nn.init.xavier_uniform_(module.weight.data, gain=kwargs.get("gain", 1.0))
        elif init_method == "xavier_normal":
            module.weight.data = nn.init.xavier_normal_(module.weight.data, gain=kwargs.get("gain", 1.0))
        elif init_method == "kaiming_uniform":
            module.weight.data = nn.init.kaiming_uniform_(
                module.weight.data,
                kwargs.get("a", 0),
                kwargs.get("mode", "fan_in"),
                kwargs.get("nonlinearity", "leaky_relu"),
            )
        elif init_method == "kaiming_normal":
            module.weight.data = nn.init.kaiming_normal_(
                module.weight.data,
                kwargs.get("a", 0),
                kwargs.get("mode", "fan_in"),
                kwargs.get("nonlinearity", "leaky_relu"),
            )
        elif init_method == "orthogonal":
            module.weight.data = nn.init.orthogonal_(module.weight.data, kwargs.get("gain", 1.0))

        if module.bias is not None:
            module.bias.data.zero_()

    elif isinstance(module, nn.Embedding):
        if init_method == "normal":
            module.weight.data.normal_(mean=kwargs.get("mean", 0.0), std=kwargs.get("std", 0.02))
        elif init_method == "xavier_uniform":
            module.weight.data = nn.init.xavier_uniform_(module.weight.data, gain=kwargs.get("gain", 1.0))
        elif init_method == "xavier_normal":
            module.weight.data = nn.init.xavier_normal_(module.weight.data, gain=kwargs.get("gain", 1.0))
        elif init_method == "kaiming_uniform":
            module.weight.data = nn.init.kaiming_uniform_(
                module.weight.data,
                kwargs.get("a", 0),
                kwargs.get("mode", "fan_in"),
                kwargs.get("nonlinearity", "leaky_relu"),
            )
        elif init_method == "kaiming_normal":
            module.weight.data = nn.init.kaiming_normal_(
                module.weight.data,
                kwargs.get("a", 0),
                kwargs.get("mode", "fan_in"),
                kwargs.get("nonlinearity", "leaky_relu"),
            )
        elif init_method == "orthogonal":
            module.weight.data = nn.init.orthogonal_(module.weight.data, kwargs.get("gain", 1.0))

        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class Freezer:
    """A utility class for freezing and unfreezing parameters in a PyTorch model.

    This class allows for freezing the parameters of a model's layers either by
    the index of the layer or by the name of the parameters. It keeps track of
    the freeze state of each parameter, allowing for selective training of model parts.

    Examples
    --------
    >>> from rich.pretty import pprint
    >>> from torch import nn
    >>> model = nn.Sequential(
    ...     nn.Conv2d(1, 20, 5, bias=False),
    ...     nn.Conv2d(20, 50, 5, bias=False),
    ...     nn.Linear(800, 500, bias=False),
    ...     nn.Linear(500, 10, bias=False),
    ... )
    >>> freezer = Freezer(model)
    >>> freezer.freeze_by_index([0, 2])
    >>> status = freezer.report_freezing()
    >>> pprint(status)
    {'0.weight': True, '1.weight': False, '2.weight': True, '3.weight': False}

    >>> assert model[0].weight.requires_grad is False
    >>> assert model[1].weight.requires_grad is True
    >>> assert model[2].weight.requires_grad is False
    >>> assert model[3].weight.requires_grad is True

    >>> freezer.freeze_by_name(["3.weight"])
    >>> status = freezer.report_freezing()
    >>> pprint(status)
    {'0.weight': True, '1.weight': False, '2.weight': True, '3.weight': True}
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize the Freezer with the model. Note given the nature of mutable
        objects, the `model` is a reference to the original model from outside the
        constructor, and any modifications to the model will be reflected inplace.
        """
        self.model = model
        self.frozen_status: Dict[str, bool] = {}
        self.update_freezing_status()

    def update_freezing_status(self) -> None:
        """
        Update the freezing status of all parameters in the model. So `True` means
        the parameter is frozen, and `False` means it is trainable.
        """
        for name, param in self.model.named_parameters():
            self.frozen_status[name] = not param.requires_grad

    def freeze_by_index(self, indices: List[int], submodule_path: str | None = None) -> None:
        """
        Freeze layers based on their index in the ordered list of model's children.

        Parameters
        ----------
        indices : List[int]
            List of indices of the layers to freeze.
        submodule_path : str, optional
            The path to the submodule to freeze in the model. For example, if the model
            has a submodule `backbone` with a Conv2d layer at index 0, then the path
            would be `backbone`.
        """

        target = self.model
        if submodule_path:
            for attr in submodule_path.split("."):
                target = getattr(target, attr)

        for idx, child in enumerate(target.children()):
            if idx in indices:
                for param in child.parameters():
                    param.requires_grad = False
        self.update_freezing_status()

    def freeze_by_name(self, names: List[str]) -> None:
        """
        Freeze layers based on the names of the parameters.

        Parameters
        ----------
        names : List[str]
            List of substrings to match in the parameter names for freezing.
            This name can be obtained by calling `model.named_parameters()`.
        """
        for name, param in self.model.named_parameters():
            if any(n in name for n in names):
                param.requires_grad = False
        self.update_freezing_status()

    def freeze_by_module(self, module: nn.Module) -> None:
        """
        Freeze all parameters in the specified module.

        Parameters
        ----------
        module : nn.Module
            The module to freeze the parameters in.
        """
        for param in module.parameters():
            param.requires_grad = False
        self.update_freezing_status()

    def report_freezing(self) -> Dict[str, bool]:
        """
        Report the freezing status of all parameters in the model.

        Returns
        -------
        Dict[str, bool]
            Dictionary with parameter names and their freezing status.
        """
        return self.frozen_status


def freeze_layers(module: nn.Module) -> None:
    """Freezes the specified number of layers in the backbone. See `Freezer` for
    a more comprehensive way to freeze layers in a model.

    Parameters
    ----------
    module : nn.Module
        The model to freeze the layers in.
    """
    for parameter in module.parameters():
        parameter.requires_grad = False


def check_optimizer_coverage(model: nn.Module, optimizer: optim.Optimizer) -> Dict[str, nn.Parameter]:
    """
    Checks if all parameters in the given model are covered by the optimizer's
    parameter groups.

    Parameters
    ----------
    model: nn.Module
        The model for which the optimizer is used.
    optimizer: optim.Optimizer
        The optimizer that should be checked.

    Returns
    -------
    uncovered_params: Dict[str, nn.Parameter]
        A dictionary containing all model parameters that are not covered by the
        optimizer's parameter groups. The keys are the parameter names and the
        values are the corresponding parameter tensors.
    """
    # 1. we gather all model parameters with names
    model_params: Dict[str, nn.Parameter] = dict(model.named_parameters())

    # 2. we gather all optimizer parameters
    optimizer_params: Set[nn.Parameter] = {param for group in optimizer.param_groups for param in group["params"]}

    # 3. we check if all parameters are covered
    uncovered_params: Dict[str, nn.Parameter] = {
        name: param for name, param in model_params.items() if param not in optimizer_params
    }
    return uncovered_params


class LayerInfo(TypedDict):
    layer_name: str
    layer_parameters: int
    dtype: str
    trainable: bool


def get_model_layer_info(model: nn.Module) -> List[LayerInfo]:
    """
    A utility function to get information about the layers in a PyTorch model.

    Parameters
    ----------
    model: nn.Module
        The model to check the dtype and trainable status.

    Returns
    -------
    List[LayerInfo]
        A list of dictionaries containing the layer name, number of parameters,
        dtype, and whether the layer is trainable or not.

    Examples
    --------
    >>> from transformers import AutoModelForSequenceClassification
    >>> model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-small")
    >>> model_layers_info = get_model_layer_info(model)
    """
    model_layers_info: List[LayerInfo] = []
    for parameter_name, parameter in model.named_parameters():
        this_layer_num_parameters = int(torch.prod(torch.tensor(parameter.size())))
        model_layers_info.append(
            {
                "layer_name": parameter_name,
                "layer_parameters": this_layer_num_parameters,
                "dtype": str(parameter.data.dtype),
                "trainable": bool(parameter.requires_grad),
            }
        )
    return model_layers_info
