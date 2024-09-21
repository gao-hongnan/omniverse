import concurrent.futures
import logging
import threading
from typing import List

import requests

from omnivault.benchmark.timer import timer

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

NUM_THREADS = 10


def get_site(url: str, session: requests.Session, thread_index: int) -> None:
    with session.get(url, timeout=30) as response:
        content = response.content
        thread_id = threading.get_ident()
        logging.debug("Thread Index %d (ID: %d), Read %d from %s", thread_index, thread_id, len(content), url)


@timer
def get_sites_with_threading(urls: List[str]) -> None:
    threads: List[threading.Thread] = []
    with requests.Session() as session:
        for index, url in enumerate(urls):
            thread = threading.Thread(
                target=get_site,
                kwargs={"url": url, "session": session, "thread_index": index},
                name=f"Thread-{index}",
            )
            threads.append(thread)

    for index, thread in enumerate(threads):
        logging.debug("Before starting thread %d (Name: %s).", index, thread.name)
        thread.start()

    # Join all threads
    for index, thread in enumerate(threads):
        logging.info("Before joining thread %d (Name: %s).", index, thread.name)
        thread.join()
        logging.info("Thread %d (Name: %s) done.", index, thread.name)


@timer
def get_sites_with_concurrent_futures(urls: List[str]) -> None:
    with requests.Session() as session, concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(get_site, url, session, i) for i, url in enumerate(urls)]
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    urls = [
        "https://www.jython.org",
        "http://olympus.realpython.org/dice",
    ] * 80

    get_sites_with_threading(urls)
    get_sites_with_concurrent_futures(urls)
