from typing import Dict, List, Union

from fastapi import FastAPI, HTTPException

app = FastAPI()

# This list of dictionaries will act as our "database"
accounts = [
    {"id": 1, "name": "John Doe", "email": "johndoe@gmail.com", "balance": 100.0},
    {"id": 2, "name": "Jane Doe", "email": "janedoe@gmail.com", "balance": 200.0},
]

# This list of dictionaries will act as our "database"
transactions = [
    {
        "id": 1,
        "account_id": 1,
        "amount": 50.0,
        "type": "deposit",
        "timestamp": "2023-08-05T14:00:00Z",
    },
    {
        "id": 2,
        "account_id": 2,
        "amount": -20.0,
        "type": "withdrawal",
        "timestamp": "2023-08-05T14:00:00Z",
    },
]


def get_transaction_by_id(transaction_id: int) -> Dict[str, Union[int, str, float]]:
    """Return the transaction with the given id."""
    for transaction in transactions:
        if transaction["id"] == transaction_id:
            return transaction
    return {}  # Return an empty dict if no transaction was found


def get_account_by_id(account_id: int) -> Dict[str, Union[int, str, float]]:
    """Return the account with the given id."""
    for account in accounts:
        if account["id"] == account_id:
            return account
    return {}  # Return an empty dict if no account was found


@app.get("/accounts")
async def get_accounts() -> List[Dict[str, Union[int, str, float]]]:
    """Return all accounts."""
    return accounts


@app.get("/accounts/{account_id}")
async def get_account(account_id: int) -> Dict[str, Union[int, str, float]]:
    """Return the account with the given id."""
    account = get_account_by_id(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    return account


@app.get("/transactions")
async def get_transactions() -> List[Dict[str, Union[int, str, float]]]:
    """Return all transactions."""
    return transactions


@app.get("/transactions/{transaction_id}")
async def get_transaction(transaction_id: int) -> Dict[str, Union[int, str, float]]:
    """Return the transaction with the given id."""
    transaction = get_transaction_by_id(transaction_id)
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return transaction


@app.post("/accounts")
async def create_account(account: Dict[str, Union[str, float]]) -> Dict[str, Union[int, str, float]]:
    """Create a new account with the given details."""
    # Here we're just adding the account to our list
    # In a real application, you would save the account to your database
    account_id = max(a["id"] for a in accounts) + 1  # Generate a new ID
    account["id"] = account_id
    accounts.append(account)
    return account


@app.put("/accounts/{account_id}")
async def update_account(
    account_id: int, account_data: Dict[str, Union[str, float]]
) -> Dict[str, Union[int, str, float]]:
    """Update an existing account with the given details."""
    # Find the account to update
    for account in accounts:
        if account["id"] == account_id:
            # Update the fields
            if "name" in account_data:
                account["name"] = account_data["name"]
            if "email" in account_data:
                account["email"] = account_data["email"]
            if "balance" in account_data:
                account["balance"] = account_data["balance"]
            return account
    raise HTTPException(status_code=404, detail="Account not found")


@app.delete("/accounts/{account_id}")
async def delete_account(account_id: int) -> Dict[str, str]:
    """Delete the account with the given id."""
    # Find the account to delete
    for account in accounts:
        if account["id"] == account_id:
            # Remove the account from the list
            accounts.remove(account)  # pylint: disable=modified-iterating-list
            return {"message": f"Account {account_id} deleted successfully"}
    raise HTTPException(status_code=404, detail="Account not found")


@app.get("/accounts/{account_id}/filter")
async def filter_accounts(
    min_balance: float,
) -> List[Dict[str, Union[int, str, float]]]:
    """Return accounts with balance greater than the specified amount."""
    filtered_accounts = [account for account in accounts if account["balance"] > min_balance]
    return filtered_accounts
