from typing import List

from pydantic import BaseModel


class AccountCreate(BaseModel):
    name: str
    email: str
    balance: float


class Account(BaseModel):
    id: int
    name: str
    email: str
    balance: float

    class Config:
        from_attributes = True
