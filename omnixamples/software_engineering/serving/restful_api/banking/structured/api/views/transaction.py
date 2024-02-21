from datetime import datetime
from typing import List, Literal, Union, cast

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from omnixamples.software_engineering.serving.restful_api.banking.structured.api.crud import (
    transaction as crud_transaction,
    account as crud_account,
)

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

router = APIRouter()


@router.get("/", response_model=List[TransactionResponse])
def read_transactions(db: Session = Depends(get_db)) -> List[TransactionResponse]:
    """READ/GET: Return all transactions."""
    transactions: List[Transaction] = crud_transaction.get_transactions(db)
    transaction_responses = []
    for transaction in transactions:
        transaction.email = crud_transaction.get_transaction_account_email(transaction)
        transaction_responses.append(transaction)

    return transaction_responses  # type: ignore[return-value]


@router.get("/{transaction_id}", response_model=TransactionResponse)
def read_transaction(transaction_id: int, db: Session = Depends(get_db)) -> TransactionResponse:
    """READ/GET: Return the transaction with the given id."""
    # SELECT * FROM transactions WHERE id = transaction_id;
    transaction: Union[Transaction, None] = crud_transaction.get_transaction(db, transaction_id)

    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")

    transaction.email = crud_transaction.get_transaction_account_email(transaction)
    return transaction


@router.post("/", response_model=TransactionCreateOrUpdateResponse)
def create_transaction(
    transaction_data: TransactionCreateRequest, db: Session = Depends(get_db)
) -> TransactionCreateOrUpdateResponse:
    """Create a new transaction with the given details."""
    # Convert timestamp string to datetime
    transaction: Transaction = Transaction(**transaction_data.model_dump(mode="python"))

    # Get the account associated with the transaction

    account: Union[Account, None] = db.query(Account).get(int(transaction.account_id))

    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    # Set the account attribute of the transaction
    transaction.account = account
    db.add(transaction)
    db.commit()
    db.refresh(transaction)
    return transaction


@router.put("/{transaction_id}", response_model=TransactionCreateOrUpdateResponse)
def update_transaction(
    transaction_id: int,
    transaction_data: TransactionUpdateRequest,
    db: Session = Depends(get_db),
) -> TransactionCreateOrUpdateResponse:
    """Update an existing transaction with the given details."""
    transaction: Union[Transaction, None] = crud_transaction.get_transaction(db, transaction_id)
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")

    for key, value in transaction_data.model_dump(mode="python").items():
        setattr(transaction, key, value)

    db.commit()
    db.refresh(transaction)
    return transaction


@router.delete("/{transaction_id}")
def delete_transaction(transaction_id: int, db: Session = Depends(get_db)) -> TransactionDeleteResponse:
    """Delete the transaction with the given id."""
    transaction: Union[Transaction, None] = crud_transaction.get_transaction(db, transaction_id)
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")

    db.delete(transaction)
    db.commit()

    return TransactionDeleteResponse(
        account_id=int(transaction.account_id),
        amount=float(transaction.amount),
        type=cast(Literal["deposit", "withdrawal"], transaction.type),
        timestamp=cast(datetime, transaction.timestamp),
        message=f"Transaction ID {transaction_id} deleted",
    )
