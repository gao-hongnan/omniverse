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

    Formulation
    -----------
    The `LayerNorm` operation for each feature dimension :math:`d=1,2,\ldots,D` in token :math:`t=1,2,\ldots,T` is defined as:

    .. math::
        \overline{\mathbf{Z}}_{td} = \frac{\mathbf{Z}_{td} - \mu_t}{\sqrt{\sigma_t^2 + \epsilon}} \cdot \gamma_d + \beta_d

    where:
        - :math:`\mathbf{Z}_{td}` is the :math:`d`-th feature of the token at position :math:`t`.
        - :math:`\mu_t` is the mean of the features for token :math:`t`.
        - :math:`\sigma_t^2` is the variance of the features for token :math:`t`.
        - :math:`\gamma_d` and :math:`\beta_d` are the scaling and shifting parameters for feature :math:`d`.
        - :math:`\epsilon` is a small constant added for numerical stability.

    The mean :math:`\mu_t` and variance :math:`\sigma_t^2` are computed across
    the features of each token, and normalization is applied independently to each token.

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
        """Initialize the learnable parameters of the layer normalization module."""
        if self.elementwise_affine:
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)  # `unbiased=False` as according to PyTorch documentation.
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
    -   :math:`\mathbf{Z}^{\mathcal{B}}`: Input tensor of shape :math:`(\mathcal{B}, T, D)`.
    -   :math:`\mathbf{Z}`: One sequence in the batch consisting of :math:`T`
        tokens, of shape :math:`(1, T, D)` or simply :math:`(T, D)` when considering
        it independently.
    -   :math:`\mathbf{Z}_t`: A single vector/token of :math:`D` features at position :math:`t`
        in the input sequence, of shape :math:`(1, D)` or simply :math:`(D,)`.
    -   :math:`Z_{td}`: :math:`d`-th element of the :math:`t`-th token vector :math:`\mathbf{Z}_t`, where
        :math:`d \in \{1, 2, \ldots, D\}`.
    -   :math:`\text{RMS}(\mathbf{Z}_t)`: Root mean square of the vector
        :math:`\mathbf{Z}_t`, applied element-wise.
    -   :math:`\gamma`: Learnable scaling parameter of shape :math:`(D,)`.
    -   :math:`\epsilon`: Small constant added to the denominator for numerical
        stability.

    Formulation
    -----------
    Given a batch of input tensor :math:`\mathbf{Z}^{\mathcal{B}}` of shape
    :math:`(\mathcal{B}, T, D)`, where :math:`\mathcal{B}` represents the batch
    size, :math:`T` the sequence length, and :math:`D` the number of features in the
    `d_model` dimension, the RMSNorm operation applies as follows for each vector
    :math:`\mathbf{Z}_t` within each sequence :math:`\mathbf{Z}` where
    :math:`\mathbf{Z} \in \mathbf{Z}^{\mathcal{B}}`. In other words, the
    normalization is applied independently on :math:`\mathbf{Z}_t` at each position
    :math:`t` within a sequence :math:`\mathbf{Z}`, and for each sequence within the
    batch.

    The RMSNorm operation for an activation vector :math:`\mathbf{Z}_t` is
    defined by:

    .. math::
        \text{RMSNorm}(\mathbf{Z}_t) = \frac{\mathbf{Z}_t}{\text{RMS}(\mathbf{Z}_t)} \odot \gamma

    Here, :math:`\text{RMS}(\mathbf{Z}_t)` computes the root mean square of the
    vector :math:`\mathbf{Z}_t`, and is defined as:

    .. math::
        \text{RMS}(\mathbf{Z}_t) = \sqrt{\frac{1}{D} \sum_{d=1}^{D} Z_{td}^2 + \epsilon}

    - :math:`Z_{td}` denotes the :math:`d`-th element of the vector :math:`\mathbf{Z}_t`,
      at time step `t`, with :math:`d` ranging from 1 to :math:`D`, which is
      the dimensionality of the feature space.
    - :math:`\gamma` is a learnable scaling parameter vector of shape :math:`(D,)`,
      allowing adaptive scaling of the normalized activation vector. It is applied
      element-wise in conjunction with the normalization factor.
    - :math:`\epsilon` is a small constant added for numerical stability, often set
      to a value like :math:`1e-5`.

    The element-wise computation for each feature :math:`Z_{td}` in the normalized
    vector :math:`\text{RMSNorm}(\mathbf{Z}_t)` can be expressed as:

    .. math::
        \text{RMSNorm}(Z_{td}) = \frac{Z_{td}}{\sqrt{\frac{1}{D} \sum_{d=1}^{D} Z_{td}^2 + \epsilon}} \cdot \gamma_d

    where :math:`\gamma_d` represents the :math:`d`-th element of the learnable
    scaling parameter vector :math:`\gamma`. This is an useful interpretation
    even though we view the normalization as being applied to the entire vector
    :math:`\mathbf{Z}_t` at once.

    .. note::
        Like `LayerNorm`, `RMSNorm` normalizes the inputs across the feature dimension
        (i.e., :math:`D`), but it uses the root mean square (RMS) of the elements
        instead of the standard deviation.

        If an input tensor :math:`\mathbf{Z}^{\mathcal{B}}` has shape :math:`(B, T, D)`, then `RMSNorm` is
        applied across the `D` dimension for each position/token :math:`\mathbf{Z}_t`
        (shape :math:`1 \times D`) in the sequence. This means that the normalization is
        applied independently for each position in the sequence and, for example, if
        :math:`B=2, T=3, D=4`, this means that the normalization is applied to
        :math:`B \times T = 6` vectors (tokens) of shape :math:`1 \times 4`.

    Attributes
    ----------
    d_model : int
        The size of the hidden dimension `d_model`.
    eps : float, optional
        A small constant added to the denominator for numerical stability. Default: 1e-5.
    device : torch.device or str, optional
        The device on which the module will be allocated. Default: None.
    dtype : torch.dtype, optional
        The data type for the module's parameters. Default: None.
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

        The `gain` parameter is initialized to ones.
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
        # x is of shape `[B, T, D]` so `D` is the last dimension
        x_root_mean_squared_BT1 = torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) / self.d_model + self.eps)
        x_normalized_BTD = x / x_root_mean_squared_BT1
        x_normalized_affine_BTD = x_normalized_BTD * self.gain
        return x_normalized_affine_BTD
