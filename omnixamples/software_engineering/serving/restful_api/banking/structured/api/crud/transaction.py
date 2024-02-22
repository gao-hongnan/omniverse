"""
By creating functions that are only dedicated to interacting with the database
(get a user or an item) independent of your path operation function, you can
more easily reuse them in multiple parts and also add unit tests for them.
"""
from datetime import datetime
from typing import List, Literal, Union, cast

from sqlalchemy.orm import Session

from omnixamples.software_engineering.serving.restful_api.banking.structured.api.database.models.account import Account
from omnixamples.software_engineering.serving.restful_api.banking.structured.api.database.models.transaction import (
    Transaction,
)
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


def get_transaction_with_email(transaction: Transaction) -> TransactionResponse:
    # we do not need to do that since I believe transaction will be coerced/converted internally to TransactionResponse
    transaction_response = cast(TransactionResponse, transaction)
    transaction_response.email = get_transaction_account_email(transaction)
    return transaction_response


def get_transactions_with_email(transactions: List[Transaction]) -> List[TransactionResponse]:
    """Return all transactions with the email of the account associated with the transaction."""
    return [get_transaction_with_email(transaction) for transaction in transactions]


def get_transaction_account_email(transaction: Transaction) -> str:
    """Get the email of the account associated with the transaction."""
    account: Account = transaction.account
    email = account.email
    return str(email)


def create_transaction(
    db: Session, transaction_data: TransactionCreateRequest
) -> Union[TransactionCreateOrUpdateResponse, None]:
    """Create a new transaction with the given details."""
    # Convert timestamp string to datetime
    transaction: Transaction = Transaction(**transaction_data.model_dump(mode="python"))

    # Get the account associated with the transaction
    account: Union[Account, None] = db.query(Account).get(int(transaction.account_id))

    if not account:
        # fmt: off
        return None # explicitly return None if account is not found so we can raise an error in the corresponding router.
        # fmt: on

    # Set the account attribute of the transaction
    transaction.account = account
    db.add(transaction)
    db.commit()
    db.refresh(transaction)
    return transaction


def update_transaction(
    db: Session, transaction_id: int, transaction_data: TransactionUpdateRequest
) -> Union[Transaction, None]:
    """Update an existing transaction with the given details."""
    transaction = get_transaction(db, transaction_id)
    if not transaction:
        return None

    for key, value in transaction_data.model_dump(mode="python").items():
        setattr(transaction, key, value)

    db.commit()
    db.refresh(transaction)
    return transaction


def delete_transaction(db: Session, transaction_id: int) -> Union[TransactionDeleteResponse, None]:
    """Delete the transaction with the given id."""
    transaction: Union[Transaction, None] = get_transaction(db, transaction_id)
    if not transaction:
        return None

    db.delete(transaction)
    db.commit()

    return TransactionDeleteResponse(
        account_id=int(transaction.account_id),
        amount=float(transaction.amount),
        type=cast(Literal["deposit", "withdrawal"], transaction.type),
        timestamp=cast(datetime, transaction.timestamp),
        message=f"Transaction ID {transaction_id} deleted",
    )
