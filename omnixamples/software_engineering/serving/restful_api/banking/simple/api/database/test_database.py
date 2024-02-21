import os
from pathlib import Path

from api.models.account import Account
from api.models.transaction import Transaction
from rich.pretty import pprint
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Get the directory of the current script
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Construct the path to the database file
DATABASE_URL = CURRENT_DIR / "database.db"

# Set up database URL as an environment variable for better flexibility

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_URL}")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

accounts = session.query(Account).all()  # Returns a list of all Account objects
for account in accounts:
    print(account.id, account.name, account.email, account.balance)

ID = 1
account = session.query(Account).filter(Account.id == ID).one()
pprint(account.__dict__)

transactions = session.query(Transaction).filter(Transaction.account_id == account.id).all()
pprint(transactions[0].__dict__)

account = session.query(Account).filter(Account.id == ID).one()
pprint(account.__dict__)

transactions = account.transactions
pprint(transactions[0].__dict__)


new_account = Account(name="John Doe", email="john@example.com", balance=100.0)
session.add(new_account)
session.commit()

account_john = session.query(Account).filter(Account.name == "John Doe").one()
pprint(account_john.__dict__)
