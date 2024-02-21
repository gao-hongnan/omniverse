from rich.pretty import pprint

from omnixamples.software_engineering.serving.restful_api.banking.structured.api.database.models.account import Account
from omnixamples.software_engineering.serving.restful_api.banking.structured.api.database.models.transaction import (
    Transaction,
)
from omnixamples.software_engineering.serving.restful_api.banking.structured.api.database.session import SessionLocal

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


# new_account = Account(name="John Doe", email="john@example.com", balance=100.0)
# session.add(new_account)
# session.commit()

# account_john = session.query(Account).filter(Account.name == "John Doe").one()
# pprint(account_john.__dict__)
