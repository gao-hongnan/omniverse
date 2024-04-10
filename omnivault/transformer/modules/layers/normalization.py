from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.types import _device, _dtype


class LayerNorm(nn.Module):
    r"""Layer Normalization (LayerNorm).

    LayerNorm is a normalization technique applied across the features of an input
    tensor. It operates on the `normalized_shape` dimensions, which are typically
    the last :math:`K` dimensions of the input tensor.

    However, in the context of transformers, we will hardcode the size of the
    tensor for clarity. To this end, given an input tensor :math:`\mathbf{X}` of shape :math:`(\mathcal{B}, T, D)`,
    where :math:`\mathcal{B}` represents the batch size, :math:`T` the sequence
    length, and :math:`D` the number of features in the `d_model` dimension, the
    `LayerNorm` operation applies normalization and affine transformation across
    the :math:`D` features dimension.

    .. note::
        It is worth noting that this implementation of `LayerNorm` is coupled
        to the transformer architecture and therefore we assumed that the
        input tensor has a shape of :math:`(\mathcal{B}, T, D)`. In practice,
        the "features" might span across channels, height, and width dimensions
        like in convolutional neural networks. Consequently, Layer Normalization
        can thus be applied across these multiple dimensions simultaneously,
        treating them collectively as the feature space for each sample.
        See `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html>`_
        for an example of how to apply `LayerNorm` across multiple dimensions.

    Notation
    --------
    - :math:`\mathbf{Z}`: Input tensor of shape :math:`(\mathcal{B}, T, D)`.
    - :math:`\mu`: Mean of the features to be normalized. Also denoted as :math:`\mathbb{E}[\mathbf{X}]`.
    - :math:`\sigma`: Standard deviation of the features to be normalized.
    - :math:`\gamma`: Learnable scaling parameter of shape :math:`(D_1, D_2, \ldots, D_n)`.
    - :math:`\beta`: Learnable bias parameter of shape :math:`(D_1, D_2, \ldots, D_n)`.
    - :math:`\epsilon`: Small constant added to the denominator for numerical stability.

    Formulation
    -----------
    LayerNorm operates on the features defined by the `normalized_shape` dimensions.
    It computes the mean and standard deviation of these features and applies
    normalization and affine transformation.

    The LayerNorm operation is defined as:

    .. math::
        \text{LayerNorm}(\mathbf{X}) = \frac{\mathbf{X} - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta

    where:
        - :math:`\mu` is the mean of the features to be normalized.
        - :math:`\sigma` is the standard deviation of the features to be normalized.
        - :math:`\gamma` is a learnable scaling parameter (if `elementwise_affine=True`).
        - :math:`\beta` is a learnable bias parameter (if `elementwise_affine=True`).
        - :math:`\epsilon` is a small constant added for numerical stability.

    The mean and standard deviation are computed across the `normalized_shape`
    dimensions of the input tensor.

    Attributes
    ----------
    normalized_shape : int or tuple of ints
        The shape of the normalized dimensions. It can be an integer if only one
        dimension is normalized or a tuple of integers for multiple dimensions.
    eps : float, optional
        A small constant added to the denominator for numerical stability. Default: 1e-5.
    elementwise_affine : bool, optional
        A boolean value indicating whether to learn per-element affine parameters
        (scaling and bias). If True, the `gamma` and `beta` parameters are learned.
        If False, no affine transformation is applied. Default: True.
    device : torch.device or str, optional
        The device on which the module will be allocated. Default: None.
    dtype : torch.dtype, optional
        The data type for the module's parameters. Default: None.
    gamma : torch.nn.Parameter or None
        The learnable scaling parameter of shape `normalized_shape` if
        `elementwise_affine=True`. None otherwise.
    beta : torch.nn.Parameter or None
        The learnable bias parameter of shape `normalized_shape` if
        `elementwise_affine=True`. None otherwise.

    Examples
    --------
    >>> layer_norm = LayerNorm(128)
    >>> x = torch.randn(2, 10, 128)  # Batch size 2, sequence length 10, feature dimension 128
    >>> output = layer_norm(x)
    >>> output.shape
    torch.Size([2, 10, 128])

    >>> layer_norm = LayerNorm((128, 256))
    >>> x = torch.randn(2, 128, 256)  # Batch size 2, normalized dimensions 128 and 256
    >>> output = layer_norm(x)
    >>> output.shape
    torch.Size([2, 128, 256])
    """
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]

    normalized_shape: Union[int, Tuple[int, ...]]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: Optional[Union[_device, str, None]] = None,
        dtype: Optional[_dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))  # type: ignore[arg-type]
            self.beta = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))  # type: ignore[arg-type]
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(
            dim=-1, keepdim=True, unbiased=False
        )  # NOTE: it is unbiased=False as according to PyTorch documentation.
        if self.elementwise_affine:
            return self.gamma * (x - mean) / (std + self.eps) + self.beta
        return (x - mean) / (std + self.eps)

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}".format(**self.__dict__)


class RMSNorm(nn.Module):
    r"""Root Mean Square Layer Normalization (RMSNorm).

    RMSNorm is a normalization technique applied position-wise to each token in a
    sequence. It operates on the feature dimension, which is the last
    dimension of the input tensor.

    For each position in the sequence, RMSNorm normalizes the vector of `d_model`
    features by dividing it by the root mean square (RMS) of its elements, adding a
    small constant `eps` for numerical stability. The normalized values are then
    scaled by a learnable `gamma` (`gain`) parameter. Thus, the only learnable weights
    in `RMSNorm` is the `gamma` parameter.

    Notation
    --------
    -   :math:`\mathcal{B}`: Batch size.
    -   :math:`T`: Sequence length.
    -   :math:`D`: Number of features in the `d_model` dimension.
    -   :math:`\mathbf{X}`: Input tensor of shape :math:`(\mathcal{B}, T, D)`.
    -   :math:`\mathbf{x}`: One sequence in the batch consisting of :math:`T`
        tokens, of shape :math:`(1, T, D)` or simply :math:`(T, D)` when considering
        it independently.
    -   :math:`\mathbf{x}_t`: A single vector/token of `D` features at position `t`
        in the input sequence, of shape :math:`(1, D)` or simply :math:`(D,)`.
    -   :math:`x_{td}`: `d`-th element of the vector :math:`\mathbf{x}_t`, where
        :math:`d \in \{1, 2, \ldots, D\}`.
    -   :math:`\text{RMS}(\mathbf{x}_t)`: Root mean square of the vector
        :math:`\mathbf{x}_t`, applied element-wise.
    -   :math:`\gamma`: Learnable scaling parameter of shape :math:`(D,)`.
    -   :math:`\epsilon`: Small constant added to the denominator for numerical
        stability.

    Formulation
    -----------
    Given an input tensor :math:`\mathbf{X}` of shape :math:`(\mathcal{B}, T, D)`, where
    :math:`\mathcal{B}` represents the batch size, :math:`T` the sequence length, and
    :math:`D` the number of features in the `d_model` dimension, the RMSNorm operation
    applies as follows for each vector :math:`\mathbf{x}_t` within :math:`\mathbf{X}`,
    at each position :math:`t` within a sequence, and for each sequence within the
    batch.

    The RMSNorm operation for an activation vector :math:`\mathbf{x}_t`, which is a
    part of the input batched tensor :math:`\mathbf{X}`, is defined by:

    .. math::
        \text{RMSNorm}(\mathbf{x}_t) = \frac{\mathbf{x}_t}{\text{RMS}(\mathbf{x}_t)} \cdot \gamma

    Here, :math:`\text{RMS}(\mathbf{x}_t)` computes the root mean square of the
    vector :math:`\mathbf{x}_t`, and is defined as:

    .. math::
        \text{RMS}(\mathbf{x}_t) = \sqrt{\frac{1}{D} \sum_{d=1}^{D} x_{td}^2 + \epsilon}

    - :math:`x_{td}` denotes the :math:`d`-th element of the vector :math:`\mathbf{x}_t`,
      with :math:`d` ranging from 1 to :math:`D`, reflecting the dimensionality of
      the feature space.
    - :math:`\gamma` is a learnable scaling parameter vector of shape :math:`(D,)`,
      allowing adaptive scaling of the normalized activation vector. It is applied
      element-wise in conjunction with the normalization factor.
    - :math:`\epsilon` is a small constant added for numerical stability, often set
      to a value like :math:`1e-5`.

    The element-wise computation for each feature :math:`x_{td}` in the normalized
    vector :math:`\text{RMSNorm}(\mathbf{x}_t)` can be expressed as:

    .. math::
        \text{RMSNorm}(x_{td}) = \frac{x_{td}}{\sqrt{\frac{1}{D} \sum_{d=1}^{D} x_{td}^2 + \epsilon}} \cdot \gamma_d

    where :math:`\gamma_d` represents the :math:`d`-th element of the learnable
    scaling parameter vector :math:`\gamma`. This is a useful interpretation
    even though we view the normalization as being applied to the entire vector
    :math:`\mathbf{x}_t` at once.

    Notes
    -----
    Like `LayerNorm`, `RMSNorm` normalizes the inputs across the feature dimension
    (i.e., `d`), but it uses the root mean square (RMS) of the elements
    instead of the standard deviation.

    In other words, if an input tensor `x` has shape `(B, T, D)`, then `RMSNorm` is
    applied across the `D` dimension for each position/token `x_t` (shape `1 x D`)
    in the sequence. This means that the normalization is applied independently
    for each position in the sequence and if `B=2, T=3, D=4`, this means that
    the normalization is applied to `B x T = 6` vectors (tokens) of shape `1 x 4`.

    Parameters
    ----------
    d : int
        The number of features in the `d` dimension.
    eps : float, optional
        A small constant added to the denominator for numerical stability. Default: 1e-5.
    device : torch.device or str, optional
        The device on which the module will be allocated. Default: None.
    dtype : torch.dtype, optional
        The data type for the module's parameters. Default: None.

    Attributes
    ----------
    gain : torch.nn.Parameter
        The learnable scaling parameter of shape `(D,)`.
    normalized_shape : Tuple[int]
        The shape of the normalized tensor, which is `(D,)`.

    Examples
    --------
    >>> rms_norm = RMSNorm(d=128)
    >>> x = torch.randn(2, 10, 128)  # Batch size 2, sequence length 10, d 128
    >>> output = rms_norm(x)
    >>> output.shape
    torch.Size([2, 10, 128])
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: Optional[Union[_device, str, None]] = None,
        dtype: Optional[_dtype] = None,
    ) -> None:
        """
        Initializes the RMSNorm module.

        Args:
            d_model (int): The number of features in the `d_model` dimension.
            eps (float, optional): A small constant added to the denominator for numerical stability. Default: 1e-5.
            device (torch.device or str, optional): The device on which the module will be allocated. Default: None.
            dtype (torch.dtype, optional): The data type for the module's parameters. Default: None.
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.gain = nn.Parameter(data=torch.empty(self.normalized_shape, **factory_kwargs))  # type: ignore[arg-type]

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets the learnable parameters of the RMSNorm module.

        The `gain` parameter is initialized to ones, following the default initialization scheme.
        """
        nn.init.ones_(self.gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the RMSNorm transformation to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, seq_len, d_model)`, where `batch_size`
            is the batch size, `seq_len` is the sequence length, and `d_model` is the
            number of features.

        Returns
        -------
        x_normalized_affine_BTD: torch.Tensor
            Normalized tensor of the same shape as the input tensor.
        """
        # x is usually of shape `[B, T, D]` so `D` is the last dimension
        x_root_mean_squared_BT1 = torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) / self.d_model + self.eps)
        x_normalized_BTD = x / x_root_mean_squared_BT1
        x_normalized_affine_BTD = x_normalized_BTD * self.gain
        return x_normalized_affine_BTD
