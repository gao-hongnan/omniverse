from __future__ import annotations

import asyncio
import inspect
import logging
import sys
import timeit
import types
from random import randint
from typing import List, cast

from rich.pretty import pprint

from omnixamples.software_engineering.concurrency_parallelism_asynchronous._async.config import parser
from omnixamples.software_engineering.concurrency_parallelism_asynchronous._async.req_http import ahttp_get, http_get

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# The highest Pokemon id
MAX_POKEMON = 898


def get_random_pokemon_name() -> str:
    this_function_name = cast(types.FrameType, inspect.currentframe()).f_code.co_name

    pokemon_id = randint(1, MAX_POKEMON)
    pokemon_url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}"
    pokemon = http_get(pokemon_url)
    logger.info("Function: %s, Pokemon: %s", this_function_name, pokemon["name"])
    return str(pokemon["name"])


async def aget_random_pokemon_name() -> str:
    this_function_name = cast(types.FrameType, inspect.currentframe()).f_code.co_name

    pokemon_id = randint(1, MAX_POKEMON)
    pokemon_url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}"
    pokemon = await ahttp_get(pokemon_url)
    logger.info("Function: %s, Pokemon: %s", this_function_name, pokemon["name"])
    return str(pokemon["name"])


async def amain(run_async: bool = True) -> List[str]:
    # synchronous call
    if not run_async:
        results = []
        time_before = timeit.default_timer()
        for _ in range(20):
            results.append(get_random_pokemon_name())  # noqa: PERF401
        logger.info("Total time (synchronous): %s", timeit.default_timer() - time_before)
        return results

    # asynchronous call
    time_before = timeit.default_timer()

    # NOTE: calling an asynchronous function like `aget_random_pokemon_name()`
    # does not immediately execute the function to completion! Instead, this
    # would create a _coroutine object_ that can be awaited later. We can easily
    # verify this by printing the type of the object returned by the function.
    # You can think of a coroutine object as a "paused" function that can be
    # resumed later. So in our case we have 20 coroutine objects.
    gathered_tasks = []
    for _ in range(20):
        gathered_tasks.append(aget_random_pokemon_name())  # noqa: PERF401

    pprint(type(gathered_tasks[0]))
    # pprint(await gathered_tasks[0])

    # NOTE: Use `asyncio.gather` to handle these coroutines concurrently where it
    # schedules the execution of all provided coroutine objects. It returns a
    # future object that eventually contains the results of all coroutines
    # when they complete.
    gathered_results = await asyncio.gather(*gathered_tasks)
    # NOTE: so what happens on a high level is that if you call 20 coroutines
    # sequentially, then perhaps the 1st coroutine would complete in 2 seconds,
    # and the 2nd coroutine would take 1 second, and so on. In this case the 2nd
    # coroutine would be _blocked_ by the 1st coroutine because it is waiting for
    # the 1st coroutine to complete before it can start, even though it may take
    # less time to complete than the 1st coroutine.

    # But if you call them concurrently using `asyncio.gather`, then even though
    # it is not entirely precise to say they are executed in parallel, you can
    # think of it as the 1st coroutine/api call is _non-blocking_ so that even
    # if it takes 2 seconds to run, since it is _awaitable_, which means
    # non-blocking wait, it yields control back to the event loop, which allows
    # the 2nd coroutine to start executing. This pattern continues until all
    # coroutines are executed. This is why the total time is less than the
    # synchronous version by quite a lot. In simple calculation, if the
    # **longest** coroutine takes 2 seconds to run, then the total time for
    # 20 coroutines would be somewhere >= 2 seconds.

    # NOTE: you can shorten the above loop by using a list comprehension below,
    # hence a bit more "pythonic" way of writing the code.
    # gathered_results = await asyncio.gather(*[aget_random_pokemon_name() for _ in range(20)])
    logger.info("Total time (asynchronous): %s", timeit.default_timer() - time_before)
    return gathered_results


if __name__ == "__main__":
    args = parser()

    # python omnixamples/software_engineering/concurrency_parallelism_asynchronous/_async/b_async_iter_vs_gather.py --sync
    # python omnixamples/software_engineering/concurrency_parallelism_asynchronous/_async/b_async_iter_vs_gather.py --async
    results = asyncio.run(amain(run_async=args.run_async))
    logger.info("Results: %s", results)
