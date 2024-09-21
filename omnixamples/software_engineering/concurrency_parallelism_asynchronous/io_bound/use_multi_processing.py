import concurrent.futures
import logging
import multiprocessing
from typing import List, Optional

import requests

from omnivault.benchmark.timer import timer

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

NUM_PROCESSES = multiprocessing.cpu_count()
logging.info("NUM_PROCESSES: %d", NUM_PROCESSES)

session: Optional[requests.Session] = None


def set_global_session() -> None:
    global session
    if not session:
        session = requests.Session()


def get_site(url: str) -> None:
    assert session is not None
    with session.get(url, timeout=30) as response:
        content = response.content
        logging.debug("Process %s, Read %d from %s", multiprocessing.current_process().name, len(content), url)


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
