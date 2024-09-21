import concurrent.futures
import logging
import multiprocessing
from logging.handlers import QueueHandler, QueueListener
from typing import List, Optional

import requests

from omnivault.benchmark.timer import timer


def setup_logging():
    queue = multiprocessing.Queue(-1)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = QueueHandler(queue)
    root.addHandler(handler)
    listener = QueueListener(queue, logging.StreamHandler())
    listener.start()
    return listener


def worker_init():
    global session
    session = requests.Session()


NUM_PROCESSES = multiprocessing.cpu_count()

session: Optional[requests.Session] = None


def get_site(url: str) -> None:
    assert session is not None
    with session.get(url, timeout=30) as response:
        content = response.content
        process_name = multiprocessing.current_process().name
        process_id = multiprocessing.current_process().pid
        logging.debug("Process %s (PID: %d), Read %d from %s", process_name, process_id, len(content), url)


@timer
def get_sites_with_multiprocessing(urls: List[str]) -> None:
    with multiprocessing.Pool(processes=NUM_PROCESSES, initializer=worker_init) as pool:
        pool.map(get_site, urls)


@timer
def get_sites_with_concurrent_futures(urls: List[str]) -> None:
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_PROCESSES, initializer=worker_init) as executor:
        futures = [executor.submit(get_site, url) for url in urls]
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    listener = setup_logging()
    logging.info("NUM_PROCESSES: %d", NUM_PROCESSES)

    urls = [
        "https://www.jython.org",
        "http://olympus.realpython.org/dice",
    ] * 80

    try:
        get_sites_with_multiprocessing(urls)
        get_sites_with_concurrent_futures(urls)
    finally:
        listener.stop()
