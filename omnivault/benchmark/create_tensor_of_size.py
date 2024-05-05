import torch

from omnivault.constants.memory import MemoryUnit


def create_tensor_of_vram_size(dtype: torch.dtype, vram_size_in_bytes: int) -> torch.Tensor:
    """Create a tensor of the specified dtype that fits in the VRAM size."""

    # 1. For example, if `dtype` is `torch.float32` then `bytes_per_element` is 4.
    #    It returns the bytes needed to store a single element of the tensor.
    if dtype.is_floating_point:
        bytes_per_element = torch.finfo(dtype).bits // MemoryUnit.BYTE
        assert (
            bytes_per_element == torch.tensor([], dtype=dtype).element_size()
        )  # TODO: may be inefficient adding this assertion
    else:
        bytes_per_element = torch.iinfo(dtype).bits // MemoryUnit.BYTE

    # 2. Simple math, we need to find the number of elements required to
    #    "consume" the target vram. For example, if we want to consume 10MB of
    #    vram and each element is 4 bytes (float32), then we need ~2.5 million
    #    elements derived from 10MB / 4 bytes per element.
    total_elements_needed = int(vram_size_in_bytes / bytes_per_element)

    # 3. Create 1D tensor with the required number of elements.
    tensor = torch.empty(total_elements_needed, dtype=dtype)
    assert tensor.size() == (
        total_elements_needed,
    ), f"Expected tensor size {total_elements_needed} but got {tensor.size()}."

    return tensor
