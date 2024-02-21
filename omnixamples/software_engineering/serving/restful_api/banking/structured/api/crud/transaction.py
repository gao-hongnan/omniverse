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


def get_transaction(db: Session, transaction_id: int) -> Union[Transaction, None]:
    """READ/GET: Return the transaction with the given id."""
    transaction: Union[Transaction, None] = db.query(Transaction).get(transaction_id)
    return transaction


def get_transactions(db: Session) -> List[Transaction]:
    """READ/GET: Return all transactions."""
    transactions: List[Transaction] = db.query(Transaction).all()
    return transactions

def get_transaction_account_email(transaction: Transaction) -> str:
    """Get the email of the account associated with the transaction."""
    account: Account = transaction.account
    email = account.email
    return str(email)