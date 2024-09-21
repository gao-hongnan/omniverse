import concurrent.futures
import logging
import multiprocessing
from typing import List, Optional

import requests

from omnivault.benchmark.timer import timer


def create_logger() -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(processName)s - %(process)d - %(levelname)s - %(message)s")
        process_id = multiprocessing.current_process().pid
        log_filename = f"process_{process_id}.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = create_logger()
NUM_PROCESSES = multiprocessing.cpu_count()
logger.info("NUM_PROCESSES: %d", NUM_PROCESSES)

session: Optional[requests.Session] = None


def set_global_session() -> None:
    global session
    if not session:
        session = requests.Session()


def get_site(url: str) -> None:
    assert session is not None
    with session.get(url, timeout=30) as response:
        content = response.content
        process_name = multiprocessing.current_process().name
        process_id = multiprocessing.current_process().pid
        logger.info("Process Name: %s, Process ID: %d, Read %d from %s", process_name, process_id, len(content), url)


@timer
def get_sites_with_multiprocessing(urls: List[str]) -> None:
    with multiprocessing.Pool(processes=NUM_PROCESSES, initializer=set_global_session) as pool:
        pool.map(get_site, urls)


@timer
def get_sites_with_concurrent_futures(urls: List[str]) -> None:
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_PROCESSES, initializer=set_global_session) as executor:
        futures = [executor.submit(get_site, url) for url in urls]
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    urls = [
        "https://www.jython.org",
        "http://olympus.realpython.org/dice",
    ] * 80

    get_sites_with_multiprocessing(urls)
    get_sites_with_concurrent_futures(urls)
