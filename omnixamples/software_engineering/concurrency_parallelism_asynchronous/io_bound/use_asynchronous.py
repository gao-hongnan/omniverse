import asyncio
import logging
from typing import List

import aiohttp

from omnivault.benchmark.timer import timer

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

CONCURRENT_REQUESTS = 10


async def get_site(url: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> None:
    async with semaphore, session.get(url, timeout=30) as response:
        content = await response.read()
        logging.debug("Coroutine %s, Read %d from %s", id(asyncio.current_task()), len(content), url)


@timer
async def get_sites_with_asyncio(urls: List[str]) -> None:
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(get_site(url, session, semaphore)) for url in urls]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    urls = [
        "https://www.jython.org",
        "http://olympus.realpython.org/dice",
    ] * 80

    asyncio.run(get_sites_with_asyncio(urls))
