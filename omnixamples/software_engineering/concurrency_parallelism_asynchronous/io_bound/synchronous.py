import logging
import time
from typing import List

import requests

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def get_site(url: str, session: requests.Session) -> None:
    with session.get(url, timeout=30) as response:
        content = response.content
        logging.debug("Read %d from %s", len(content), url)


def read_sites(urls: List[str]) -> None:
    with requests.Session() as session:
        for url in urls:
            get_site(url, session)


if __name__ == "__main__":
    urls = [
        "https://www.jython.org",
        "http://olympus.realpython.org/dice",
    ] * 60
    start_time = time.time()
    read_sites(urls)
    duration = time.time() - start_time
    logging.info("Downloaded %d in %d seconds", len(urls), duration)
