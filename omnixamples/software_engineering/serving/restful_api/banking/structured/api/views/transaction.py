from datetime import datetime
from typing import Dict, List, Union

from fastapi import APIRouter, Depends, HTTPException
from rich.pretty import pprint
from sqlalchemy.orm import Session

from omnixamples.software_engineering.serving.restful_api.banking.structured.api.database.models.account import Account
from omnixamples.software_engineering.serving.restful_api.banking.structured.api.database.models.transaction import (
    Transaction,
)
from omnixamples.software_engineering.serving.restful_api.banking.structured.api.database.session import get_db
from omnixamples.software_engineering.serving.restful_api.banking.structured.api.schemas.transaction import (
    TransactionCreateOrUpdateResponse,
    TransactionCreateRequest,
    TransactionResponse,
    TransactionUpdateRequest,
)

router = APIRouter()


@router.get("/", response_model=List[TransactionResponse])
async def get_transactions(db: Session = Depends(get_db)) -> List[TransactionResponse]:
    """Return all transactions."""
    transactions = db.query(Transaction).all()
    transaction_responses = []
    for transaction in transactions:
        pprint(transaction.id)
        print(transaction.account)
        account: Account = transaction.account
        email = account.email
        transaction.email = email
        transaction_responses.append(transaction)

    return transaction_responses


@router.get("/{transaction_id}", response_model=TransactionResponse)
async def get_transaction(transaction_id: int, db: Session = Depends(get_db)) -> TransactionResponse:
    """Return the transaction with the given id."""
    # SELECT * FROM transactions WHERE id = transaction_id;
    transaction: Union[Transaction, None] = db.query(Transaction).get(transaction_id)

    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")

    account: Account = transaction.account  # we can do this because of the relationship defined in the models
    email = account.email
    transaction.email = email
    return transaction


@router.post("/", response_model=TransactionCreateOrUpdateResponse)
def create_transaction(
    transaction_data: TransactionCreateRequest, db: Session = Depends(get_db)
) -> TransactionCreateOrUpdateResponse:
    """Create a new transaction with the given details."""
    # Convert timestamp string to datetime
    print(transaction_data.timestamp)
    transaction = Transaction(**transaction_data.model_dump(mode="python"))
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
    transaction = db.query(Transaction).get(transaction_id)
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")

    for key, value in transaction_data.model_dump(mode="python").items():
        setattr(transaction, key, value)

    db.commit()
    db.refresh(transaction)
    return transaction


@router.delete("/{transaction_id}")
def delete_transaction(transaction_id: int, db: Session = Depends(get_db)) -> Dict[str, str]:
    """Delete the transaction with the given id."""
    transaction = db.query(Transaction).get(transaction_id)
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")

    db.delete(transaction)
    db.commit()

    return {"message": f"Transaction {transaction_id} deleted successfully"}
