from enum import Enum


class TorchDtype(Enum):
    """Enum for PyTorch data types with memory size in bytes."""

    FLOAT32 = ("torch.float32", 4)
    FLOAT64 = ("torch.float64", 8)
    FLOAT16 = ("torch.float16", 2)
    BFLOAT16 = ("torch.bfloat16", 2)
    INT8 = ("torch.int8", 1)
    INT16 = ("torch.int16", 2)
    INT32 = ("torch.int32", 4)
    INT64 = ("torch.int64", 8)
    UINT8 = ("torch.uint8", 1)
    BOOL = ("torch.bool", 1)

    def __init__(self, dtype_str: str, size_in_bytes: int) -> None:
        self.dtype_str = dtype_str
        self.size_in_bytes = size_in_bytes

    def __str__(self) -> str:
        """Return the string representation of the enum value for dtype."""
        return self.dtype_str

    def get_size_in_bytes(self) -> int:
        """Return the size in bytes of the dtype."""
        return self.size_in_bytes
