from pydantic import BaseModel, Field, PositiveFloat


class AccountBase(BaseModel):
    name: str = Field(..., description="The name of the account.")
    email: str = Field(..., description="The email of the account, must be unique.")
    balance: PositiveFloat = Field(..., ge=0, description="The balance of the account.")

    class Config:
        from_attributes = True


class AccountCreateRequest(AccountBase):
    ...


class AccountUpdateRequest(AccountBase):
    ...


class AccountResponse(AccountBase):
    id: int


class AccountCreateOrUpdateResponse(AccountBase):
    id: int


class AccountDeleteResponse(AccountBase):
    message: str
