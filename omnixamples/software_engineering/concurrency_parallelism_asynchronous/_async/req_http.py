from __future__ import annotations

import asyncio
from typing import Dict, List, Union, Any

import requests

JSONObject = Dict[str, Any]


def http_get(url: str) -> JSONObject:
    response = requests.get(url)
    return response.json()  # type: ignore[no-any-return]


async def ahttp_get(url: str) -> JSONObject:
    return await asyncio.to_thread(http_get, url)
