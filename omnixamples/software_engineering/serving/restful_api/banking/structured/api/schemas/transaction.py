from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TransactionBase(BaseModel):
    account_id: int = Field(
        ..., ge=1, description="The id of the account the transaction is associated with, account's ID starts from 1."
    )
    amount: float = Field(..., ge=0, description="The amount of the transaction.")
    type: Literal["deposit", "withdrawal"] = Field(
        ..., description="The type of the transaction, will throw an error if not 'deposit' or 'withdrawal'."
    )
    timestamp: datetime = Field(..., description="The timestamp of the transaction.", examples=["2021-08-01T12:00:00"])

    class Config:
        from_attributes = True


# Used for creating a new transaction
class TransactionCreateRequest(TransactionBase):
    ...


# Used for updating an existing transaction
class TransactionUpdateRequest(TransactionBase):
    ...


# Used for response when a transaction is created or updated
class TransactionCreateOrUpdateResponse(TransactionBase):
    id: int


# Used for response when a transaction is retrieved
class TransactionResponse(TransactionBase):
    id: int
    email: str


class TransactionDeleteResponse(TransactionBase):
    message: str
