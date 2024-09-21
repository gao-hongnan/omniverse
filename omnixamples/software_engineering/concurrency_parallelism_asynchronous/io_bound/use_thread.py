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
        logging.debug("Thread %s, Read %d from %s", thread_index, len(content), url)


@timer
def get_sites_with_threading(urls: List[str]) -> None:
    threads: List[threading.Thread] = []
    with requests.Session() as session:
        for index, url in enumerate(urls):
            thread = threading.Thread(
                target=get_site,
                kwargs={"url": url, "session": session, "thread_index": index},
            )
            threads.append(thread)

    for index, thread in enumerate(threads):
        logging.debug("before starting thread %d.", index)
        thread.start()

    for index, thread in enumerate(threads):
        logging.info("before joining thread %d.", index)
        thread.join()
        logging.info("thread %d done", index)


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
