from typing import Union

from pydantic import BaseModel, Field

__all__ = ["GeneratorConfig"]


class GeneratorConfig(BaseModel):
    max_tokens: int = Field(default=1000)
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)
    greedy: bool = Field(default=False)
    top_k: Union[int, None] = Field(default=None)
    top_p: Union[float, None] = Field(default=None)
