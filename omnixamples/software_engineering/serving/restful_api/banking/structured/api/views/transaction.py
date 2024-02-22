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
    transaction_responses: List[TransactionResponse] = crud_transaction.get_transactions_with_email(transactions)
    return transaction_responses


@router.get("/{transaction_id}", response_model=TransactionResponse)
def read_transaction(transaction_id: int, db: Session = Depends(get_db)) -> TransactionResponse:
    """READ/GET: Return the transaction with the given id."""
    # SELECT * FROM transactions WHERE id = transaction_id;
    transaction: Union[Transaction, None] = crud_transaction.get_transaction(db, transaction_id)

    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")

    transaction_response: TransactionResponse = crud_transaction.get_transaction_with_email(transaction)
    return transaction_response


@router.post("/", response_model=TransactionCreateOrUpdateResponse)
def create_transaction(
    transaction_data: TransactionCreateRequest, db: Session = Depends(get_db)
) -> TransactionCreateOrUpdateResponse:
    """Create a new transaction with the given details."""
    transaction = crud_transaction.create_transaction(db, transaction_data)
    if not transaction:
        raise HTTPException(status_code=404, detail="Account not found")
    return transaction


@router.put("/{transaction_id}", response_model=TransactionCreateOrUpdateResponse)
def update_transaction(
    transaction_id: int,
    transaction_data: TransactionUpdateRequest,
    db: Session = Depends(get_db),
) -> TransactionCreateOrUpdateResponse:
    """Update an existing transaction with the given details."""
    update_transaction = crud_transaction.update_transaction(db, transaction_id, transaction_data)
    if not update_transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return update_transaction

@router.delete("/{transaction_id}")
def delete_transaction(transaction_id: int, db: Session = Depends(get_db)) -> TransactionDeleteResponse:
    """Delete the transaction with the given id."""
    transaction_to_delete = crud_transaction.delete_transaction(db, transaction_id)
    if not transaction_to_delete:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return transaction_to_delete
