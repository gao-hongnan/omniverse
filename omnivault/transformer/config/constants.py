from typing import List

from pydantic import BaseModel, Field


class MaybeConstant(BaseModel):
    NUM_DIGITS: int = Field(default=2, description="The number of digits in the sequence.")
    TOKENS: List[str] = Field(
        default=[
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "+",
            "*",
            "-",
            "=",
            "<BOS>",
            "<EOS>",
            "<PAD>",
            "<UNK>",
        ],
        description="The tokens in the vocabulary.",
    )
