"""
By creating functions that are only dedicated to interacting with the database
(get a user or an item) independent of your path operation function, you can
more easily reuse them in multiple parts and also add unit tests for them.
"""
from typing import List, Union

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from omnixamples.software_engineering.serving.restful_api.banking.structured.api.database.models.account import Account
from omnixamples.software_engineering.serving.restful_api.banking.structured.api.database.models.transaction import (
    Transaction,
)
from omnixamples.software_engineering.serving.restful_api.banking.structured.api.database.session import get_db
from omnixamples.software_engineering.serving.restful_api.banking.structured.api.schemas.transaction import (
    TransactionCreateOrUpdateResponse,
    TransactionCreateRequest,
    TransactionDeleteResponse,
    TransactionResponse,
    TransactionUpdateRequest,
)

def get_account(db: Session, account_id: int) -> Union[Account, None]:
    """READ/GET: Return the account with the given id."""
    account: Union[Account, None] = db.query(Account).get(account_id)
    return account