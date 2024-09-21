"""With reference to effective python book chapter 54.
Ref: https://github.com/bslatkin/effectivepython/blob/master/example_code/item_54.py
"""
import logging
import threading
from threading import Barrier
from typing import List

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

NUM_THREADS = 5
BARRIER = Barrier(NUM_THREADS)


class Counter:
    def __init__(self) -> None:
        self.count = 0

    def increment(self, offset: int) -> None:
        self.count += offset


def worker(thread_index: int, total_iterations: int, counter: Counter) -> None:
    """The barrier is used to synchronize the threads so that they all start counting
    at the same time. This makes it easier to get a race condition since we wait for
    the other threads to start else in the loop we always have an order that the
    first thread likely starts first and then the second and so on.
    """
    BARRIER.wait()
    logging.debug("Thread %s, starting", thread_index)
    for _ in range(total_iterations):
        counter.increment(1)


if __name__ == "__main__":
    total_iterations = 10**6
    counter = Counter()

    threads: List[threading.Thread] = []
    for index in range(NUM_THREADS):
        thread = threading.Thread(target=worker, args=(index, total_iterations, counter))
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    expected = total_iterations * NUM_THREADS
    found = counter.count
    print(f"Counter should be {expected}, got {found}")
