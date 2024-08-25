from __future__ import annotations

import asyncio
import logging
import sys
from random import randint
from typing import Any, Dict, cast

from omnixamples.software_engineering.concurrency_parallelism_asynchronous._async.req_http import ahttp_get

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# The highest Pokemon ID
MAX_POKEMON = 898


async def fetch_pokemon_detail(pokemon_id: int) -> Dict[str, Any]:
    pokemon_url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}"
    pokemon_details = await ahttp_get(pokemon_url)
    logger.info("Pokemon Details: %s", pokemon_details["name"])
    return pokemon_details


async def fetch_pokemon_capture_rate(pokemon_id: int) -> float:
    species_url = f"https://pokeapi.co/api/v2/pokemon-species/{pokemon_id}"
    species_info = await ahttp_get(species_url)
    capture_rate = species_info["capture_rate"]
    logger.info("Capture Rate: %s", capture_rate)
    return cast(float, capture_rate)


async def fetch_pokemon_color(pokemon_id: int) -> str:
    species_url = f"https://pokeapi.co/api/v2/pokemon-species/{pokemon_id}"
    species_info = await ahttp_get(species_url)
    color = species_info["color"]["name"]
    logger.info("Color: %s", color)
    return cast(str, color)


async def aget_pokemon_data() -> None:
    pokemon_id = randint(1, MAX_POKEMON)
    details_task = fetch_pokemon_detail(pokemon_id)
    capture_rate_task = fetch_pokemon_capture_rate(pokemon_id)
    color_task = fetch_pokemon_color(pokemon_id)

    pokemon_details, capture_rate, color = await asyncio.gather(details_task, capture_rate_task, color_task)

    logger.info("Pokemon Name: %s || Capture Rate: %s || Color: %s", pokemon_details["name"], capture_rate, color)


async def aget_pokemon_datas(n: int) -> None:
    gathered_details_tasks = []
    gathered_capture_rate_tasks = []
    gathered_color_tasks = []
    for _ in range(n):
        pokemon_id = randint(1, MAX_POKEMON)
        details_task = fetch_pokemon_detail(pokemon_id)
        capture_rate_task = fetch_pokemon_capture_rate(pokemon_id)
        color_task = fetch_pokemon_color(pokemon_id)
        gathered_details_tasks.append(details_task)
        gathered_capture_rate_tasks.append(capture_rate_task)
        gathered_color_tasks.append(color_task)


async def main() -> None:
    await aget_pokemon_data()


if __name__ == "__main__":
    asyncio.run(main())
