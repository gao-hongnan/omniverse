from __future__ import annotations

import asyncio
import logging
import sys
from random import randint

from omnivault.benchmark.timer import timer
from omnixamples.software_engineering.concurrency_parallelism_asynchronous._async.config import parser
from omnixamples.software_engineering.concurrency_parallelism_asynchronous._async.req_http import (
    JSONObject,
    ahttp_get,
    http_get,
)

__metadata__ = "Credits: https://github.com/ArjanCodes/2022-asyncio/tree/main"

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# The highest Pokemon id
MAX_POKEMON = 898


def get_pokemon(pokemon_id: int) -> JSONObject:
    pokemon_url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}"
    return http_get(pokemon_url)


async def aget_pokemon(pokemon_id: int) -> JSONObject:
    pokemon_url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}"
    return await ahttp_get(pokemon_url)


@timer
async def amain() -> None:
    pokemon_id = randint(1, MAX_POKEMON)
    pokemon = await aget_pokemon(pokemon_id + 1)
    logger.info(pokemon["name"])


@timer
def main() -> None:
    pokemon_id = randint(1, MAX_POKEMON)
    pokemon = get_pokemon(pokemon_id)
    logger.info(pokemon["name"])


if __name__ == "__main__":
    args = parser()

    if args.run_async:
        logger.info("Running asynchronously")
        asyncio.run(amain())
    else:
        logger.info("Running synchronously")
        main()
