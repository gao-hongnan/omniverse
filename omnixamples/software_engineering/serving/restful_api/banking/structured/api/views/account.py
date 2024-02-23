from typing import List, Union, cast

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from omnixamples.software_engineering.serving.restful_api.banking.structured.api.conf.constants import StatusMessage
from omnixamples.software_engineering.serving.restful_api.banking.structured.api.crud import account as crud_account
from omnixamples.software_engineering.serving.restful_api.banking.structured.api.database.models.account import Account
from omnixamples.software_engineering.serving.restful_api.banking.structured.api.database.session import get_db
from omnixamples.software_engineering.serving.restful_api.banking.structured.api.schemas.account import (
    AccountCreateOrUpdateResponse,
    AccountCreateRequest,
    AccountDeleteResponse,
    AccountResponse,
    AccountUpdateRequest,
)

router = APIRouter()


@router.get("/", response_model=List[AccountResponse])
def get_accounts(db: Session = Depends(get_db)) -> List[AccountResponse]:
    """Return all accounts."""
    accounts: List[Account] = crud_account.get_accounts(db)
    return cast(List[AccountResponse], accounts)


@router.get("/{account_id}", response_model=AccountResponse)
def get_account(account_id: int, db: Session = Depends(get_db)) -> AccountResponse:
    """Return the account with the given id."""
    # SELECT * FROM accounts WHERE id = account_id;
    account: Union[Account, None] = crud_account.get_account(db, account_id)
    if not account:
        raise HTTPException(status_code=404, detail=StatusMessage.ACCOUNT_NOT_FOUND.value.format(str(account_id)))
    return account


@router.post("/", response_model=AccountCreateOrUpdateResponse)
def create_account(account_data: AccountCreateRequest, db: Session = Depends(get_db)) -> AccountCreateOrUpdateResponse:
    """Create a new account with the given details."""
    account: Account = Account(**account_data.model_dump(mode="python"))
    db.add(account)
    db.commit()
    db.refresh(account)
    return account


@router.put("/{account_id}", response_model=AccountCreateOrUpdateResponse)
def update_account(
    account_id: int, account_data: AccountUpdateRequest, db: Session = Depends(get_db)
) -> AccountCreateOrUpdateResponse:
    """Update an existing account with the given details.

    Raises IntegrityError because we should not allow user to have
    the same email.
    """
    try:
        account: Union[Account, None] = db.query(Account).get(account_id)
        if not account:
            raise HTTPException(status_code=404, detail=StatusMessage.ACCOUNT_NOT_FOUND.value.format(str(account_id)))

        for key, value in account_data.model_dump(mode="python").items():
            setattr(account, key, value)
        db.commit()
        db.refresh(account)
        return account
    except IntegrityError as err:
        raise HTTPException(400, detail=f"Integrity error: {str(err)}") from err


@router.delete("/{account_id}")
def delete_account(account_id: int, db: Session = Depends(get_db)) -> AccountDeleteResponse:
    """Delete the account with the given id."""
    account: Union[Account, None] = db.query(Account).get(account_id)
    if not account:
        raise HTTPException(status_code=404, detail=StatusMessage.ACCOUNT_NOT_FOUND.value.format(str(account_id)))

    db.delete(account)
    db.commit()

    return AccountDeleteResponse(
        name=str(account.name),
        email=str(account.email),
        balance=float(account.balance),
        message=f"Account with id {account_id} has been deleted",
    )
