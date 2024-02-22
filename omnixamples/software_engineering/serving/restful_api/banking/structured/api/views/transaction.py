from typing import List, Union

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from omnixamples.software_engineering.serving.restful_api.banking.structured.api.crud import (
    transaction as crud_transaction,
)
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
from omnixamples.software_engineering.serving.restful_api.banking.structured.api.conf.constants import StatusMessage

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
        raise HTTPException(status_code=404, detail=StatusMessage.TRANSACTION_NOT_FOUND.value.format(str(transaction_id)))

    transaction_response: TransactionResponse = crud_transaction.get_transaction_with_email(transaction)
    return transaction_response


@router.post("/", response_model=TransactionCreateOrUpdateResponse)
def create_transaction(
    transaction_data: TransactionCreateRequest, db: Session = Depends(get_db)
) -> TransactionCreateOrUpdateResponse:
    """Create a new transaction with the given details."""
    transaction = crud_transaction.create_transaction(db, transaction_data)
    if not transaction:
        # even though it is account id, but since transaction's account_id is a foreign key to account's id, it is same.
        raise HTTPException(status_code=404, detail=StatusMessage.ACCOUNT_NOT_FOUND.value.format(str(transaction_data.account_id)))
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
        raise HTTPException(status_code=404, detail=StatusMessage.TRANSACTION_NOT_FOUND.value.format(str(transaction_id)))
    return update_transaction


@router.delete("/{transaction_id}")
def delete_transaction(transaction_id: int, db: Session = Depends(get_db)) -> TransactionDeleteResponse:
    """Delete the transaction with the given id."""
    transaction_to_delete = crud_transaction.delete_transaction(db, transaction_id)
    if not transaction_to_delete:
        raise HTTPException(status_code=404, detail=StatusMessage.TRANSACTION_NOT_FOUND.value.format(str(transaction_id)))
    return transaction_to_delete
