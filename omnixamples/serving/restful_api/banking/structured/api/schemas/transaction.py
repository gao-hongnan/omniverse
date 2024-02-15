from datetime import datetime
from typing import Optional

from pydantic import BaseModel, validator

# pylint: disable=no-self-argument


# Used for creating a new transaction
class TransactionCreateRequest(BaseModel):
    account_id: int
    amount: float
    type: str
    timestamp: str

    # convert timestamp
    @validator("timestamp", pre=True)
    def format_timestamp(cls, v):
        if isinstance(v, datetime):
            return v.isoformat()
        return v


# Used for updating an existing transaction
class TransactionUpdateRequest(BaseModel):
    account_id: Optional[int] = None
    amount: Optional[float] = None
    type: Optional[str] = None
    timestamp: Optional[str] = None

    # convert timestamp
    @validator("timestamp", pre=True)
    def format_timestamp(cls, v):
        if isinstance(v, datetime):
            return v.isoformat()
        return v


# Used for response when a transaction is created or updated
class TransactionCreateOrUpdateResponse(BaseModel):
    id: int
    account_id: int
    amount: float
    type: str
    timestamp: str

    class Config:
        from_attributes = True

    # convert timestamp
    @validator("timestamp", pre=True)
    def format_timestamp(cls, v):
        if isinstance(v, datetime):
            return v.isoformat()
        return v


# Used for response when a transaction is retrieved
class TransactionResponse(BaseModel):
    id: int
    account_id: int
    amount: float
    type: str
    timestamp: str

    class Config:
        from_attributes = True

    # convert timestamp
    @validator("timestamp", pre=True)
    def format_timestamp(cls, v):
        if isinstance(v, datetime):
            return v.isoformat()
        return v
