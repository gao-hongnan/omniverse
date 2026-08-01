from typing import Annotated

from pydantic import BaseModel, Field

__all__ = ["GeneratorConfig"]


class GeneratorConfig(BaseModel):
    max_tokens: int = 1000
    temperature: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    greedy: bool = False
    top_k: int | None = None
    top_p: float | None = None
