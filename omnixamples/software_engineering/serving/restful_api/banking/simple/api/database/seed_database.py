"""
The term "seed" in the context of databases refers to the process of populating
a database with initial set of data.

Just like in gardening, where you plant a "seed" to grow a plant, in the realm
of databases, you "seed" a database to populate it with data. The data you seed
can be used for testing, development, or even initial data in production.

In many cases, especially during development and testing, developers seed the
database with fake or sample data. This allows them to test various features of
the application and ensure it behaves as expected with that data.

So when you hear the term "database seeding" or "seed data", it refers to the
initial set of data that you're using to populate the database.
"""

import os
import random
from pathlib import Path

from faker import Faker
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from omnivault.utils.reproducibility.seed import seed_all
from omnixamples.software_engineering.serving.restful_api.banking.simple.api.models import Base
from omnixamples.software_engineering.serving.restful_api.banking.simple.api.models.account import Account
from omnixamples.software_engineering.serving.restful_api.banking.simple.api.models.transaction import Transaction

seed_all(42, seed_torch=False)

# Get the directory of the current script
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Construct the path to the database file
DATABASE_URL = CURRENT_DIR / "database.db"

# Set up database URL as an environment variable for better flexibility

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_URL}")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

fake = Faker()


def create_fake_account() -> Account:
    """Create and return a fake Account."""
    return Account(
        name=fake.name(),
        email=fake.email(),
        balance=random.uniform(1000, 5000),
    )


def create_fake_transaction(account_id: int) -> Transaction:
    """Create and return a fake Transaction for a given account_id."""
    return Transaction(
        account_id=account_id,
        amount=random.uniform(50, 500),
        type=random.choice(["deposit", "withdrawal"]),
        timestamp=fake.date_time_this_year(),
    )


def seed_database(num_accounts: int = 50, min_transactions: int = 1, max_transactions: int = 5) -> None:
    """Seed the database with fake accounts and transactions."""
    Base.metadata.create_all(bind=engine)

    for _ in range(num_accounts):
        with SessionLocal() as session:
            try:
                # Create a new fake account
                new_account = create_fake_account()
                session.add(new_account)
                session.flush()  # Ensure the account is persisted so we can get its id

                # Create a few random transactions for the account
                for _ in range(random.randint(min_transactions, max_transactions)):
                    new_transaction = create_fake_transaction(int(new_account.id))
                    session.add(new_transaction)

                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                print(f"An error occurred while seeding the database: {e}")


if __name__ == "__main__":
    seed_database()
