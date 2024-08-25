import os
from pathlib import Path
from typing import List

from sqlalchemy import create_engine, exists
from sqlalchemy.orm import Session, sessionmaker

from omnixamples.software_engineering.serving.restful_api.banking.simple.api.models.account import Account
from omnixamples.software_engineering.serving.restful_api.banking.simple.api.models.transaction import Transaction

CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATABASE_URL = CURRENT_DIR / "database.db"
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_URL}")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
session: Session = SessionLocal()


def test_create_and_retrieve_account() -> None:
    new_account: Account = Account(name="John Doe", email="john@example.com", balance=100.0)
    email_exists = session.query(exists().where(Account.email == new_account.email)).scalar()

    if not email_exists:
        session.add(new_account)
        session.commit()

    retrieved_account: Account = session.query(Account).filter(Account.name == "John Doe").one()
    assert retrieved_account.name == "John Doe"
    assert retrieved_account.email == "john@example.com"
    assert retrieved_account.balance == 100.0


def test_retrieve_all_accounts() -> None:
    accounts: List[Account] = [
        Account(name="Alice", email="alice@example.com", balance=200.0),
        Account(name="Bob", email="bob@example.com", balance=300.0),
    ]
    session.add_all(accounts)
    session.commit()

    all_accounts: List[Account] = session.query(Account).all()
    assert len(all_accounts) >= 2
    assert any(account.name == "Alice" for account in all_accounts)
    assert any(account.name == "Bob" for account in all_accounts)


def test_account_transactions() -> None:
    account: Account = Account(name="Charlie", email="charlie@example.com", balance=400.0)
    session.add(account)
    session.commit()

    transaction: Transaction = Transaction(account_id=account.id, amount=50.0, type="deposit")
    session.add(transaction)
    session.commit()

    retrieved_account: Account = session.query(Account).filter(Account.name == "Charlie").one()
    assert len(retrieved_account.transactions) == 1
    assert retrieved_account.transactions[0].amount == 50.0
    assert retrieved_account.transactions[0].type == "deposit"


if __name__ == "__main__":
    test_create_and_retrieve_account()
    test_retrieve_all_accounts()
    test_account_transactions()
