import logging
import logging.handlers

from omnivault.constants.memory import MemoryUnit

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def oom_1(block_size: int = MemoryUnit.MiB) -> None:
    """Simple experiment to induce Kill.

    1KiB = 1024B
    1MiB = 1024KB = (1024 * 1024)B
    1 letter such as 'a' is considered 1 byte approx so creating a string of
    `block_size` will consume `block_size` bytes of memory (i.e. if `block_size`
    is 1MiB then it consumes 1024 * 1024 bytes of memory). Means if block size
    is 1024KiB then large list will append "a" 1024 times which is 1024 bytes.
    So if you append such a string another time, then large list is length
    of 2 and consumes 2048 bytes of memory.

    You eventually run out of memory and the process is killed by the OS.
    Use `sudo dmesg` to see the logs. Sample out of memory log:
    Out of memory: Killed process 15021 (python) total-vm:15978908kB, anon-rss:15778968kB, file-rss:0kB, shmem-rss:0kB, UID:1001 pgtables:31024kB oom_score_adj:0
    """
    large_list = []
    letter_approx_1_byte = "a"
    total_bytes = 0

    while True:
        large_list.append(letter_approx_1_byte * block_size)
        total_bytes += block_size
        LOGGER.info(f"Allocated approximately {MemoryUnit.convert(total_bytes, MemoryUnit.BYTE, MemoryUnit.GiB)} GB")


def oom_2(loops: int = 1000000000, log_every: int = 10000) -> None:
    """Simple experiment to induce Kill."""
    locals_dict = {}
    for x in range(loops):
        if x % log_every == 0:
            LOGGER.info(f"Allocated approximately {MemoryUnit.convert(x, MemoryUnit.BYTE, MemoryUnit.GiB)} GB")
        locals_dict[f"x{x}"] = x
