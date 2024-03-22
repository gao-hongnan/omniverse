# Application: Designing a RESTful Banking API with FastAPI and SQLAlchemy

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)

```{contents}
:local:
```

## Introduction

In the rapidly evolving landscape of technology, the demand for efficient,
reliable, and scalable solutions is ever increasing. The banking sector, being
one of the most critical parts of the economy, is no exception to this trend.
The days of traditional banking, characterized by long queues and time-consuming
processes, are quickly being replaced by digital solutions that offer
convenience, speed, and broad accessibility. This digital transformation has
necessitated the development of robust and reliable banking systems that can
handle a wide range of operations, from account management to complex
transactions.

One of the key components of a modern banking system is an
[**Application Programming Interface (API)**](https://en.wikipedia.org/wiki/API),
which serves as the backbone for enabling internal and external communication
and integration. In this context, Python's
[**FastAPI**](https://fastapi.tiangolo.com/) framework has emerged as a popular
choice for creating RESTful APIs, thanks to its simplicity, speed, and
scalability.

## Problem Statement

Modern banking systems require robust and secure APIs to manage accounts and
facilitate various transactions. These APIs need to handle a variety of
operations, ranging from account creation to complex transactions such as
deposits, withdrawals, and transfers between accounts. Each of these operations
involves specific requests and responses that must be accurately processed to
ensure reliable functionality and data integrity.

In our simplified banking API system, we need to design and implement the
following operations:

1. **Account Creation**: A request to create a new account should include
   necessary information such as the account holder's name and email. The
   response should confirm successful account creation and return the details of
   the created account, including a unique account ID.

2. **Account Retrieval**: A request to retrieve an account should include the
   unique account ID. The response should return the details of the requested
   account, or an error if no such account exists.

3. **Account Update**: A request to update an account should include the unique
   account ID and the new data. The response should confirm successful account
   update and return the details of the updated account, or an error if no such
   account exists.

4. **Account Deletion**: A request to delete an account should include the
   unique account ID. The response should confirm successful account deletion,
   or an error if no such account exists.

5. **Deposit**: A request to deposit money should include the account ID and the
   amount to be deposited. The response should confirm the successful deposit
   and return the details of the transaction.

6. **Withdrawal**: A request to withdraw money should include the account ID and
   the amount to be withdrawn. The response should confirm the successful
   withdrawal and return the details of the transaction, or an error if the
   account has insufficient funds.

7. **Transfer**: A request to transfer money should include the source account
   ID, the destination account ID, and the amount to be transferred. The
   response should confirm the successful transfer and return the details of the
   transaction, or an error if the source account has insufficient funds or if
   either account does not exist.

## Models

In the context of designing a RESTful API using FastAPI and SQLAlchemy, models
refer to Python classes that represent the database entities (or tables) in our
system. These models will serve as an abstraction layer for interacting with the
database. They define the schema of the tables, the data types of each column,
and any constraints such as primary and foreign keys.

FastAPI uses Pydantic models to validate incoming data, while SQLAlchemy models
represent the database schema and enable interaction with the database. In the
context of our banking system, we have two models: `Account` and `Transaction`.

1. **Account**: This model represents a bank account. It might include fields
   like `id` (a unique identifier for each account), `name` (the account
   holder's name), `email` (the account holder's email), and `balance` (the
   current balance of the account).

2. **Transaction**: This model represents a transaction that can occur on an
   account. It might include fields like `id` (a unique identifier for each
   transaction), `account_id` (a foreign key that links the transaction to an
   account), `amount` (the amount of money involved in the transaction), `type`
   (the type of transaction, such as deposit, withdrawal, or transfer), and
   `timestamp` (the date and time when the transaction occurred).

In our application, we're using SQLAlchemy, an Object-Relational Mapping (ORM)
library for Python, to define these models. With SQLAlchemy, each model is
represented as a Python class, and instances of these classes can be directly
mapped to rows in a database table. This allows us to interact with our database
in a more Pythonic and intuitive way, as we can work with objects and classes
instead of writing raw SQL queries.

### Account Model

The `Account` model represents a bank account. It has four fields: `id`, `name`,
`email`, and `balance`, along with a `transactions` relationship field. The
schema is represented in table format below:

| Field        | Type     | Constraint           |
| ------------ | -------- | -------------------- |
| id           | Integer  | Primary Key, Indexed |
| name         | String   | Indexed              |
| email        | String   | Unique, Indexed      |
| balance      | Float    | Default = 0.0        |
| transactions | Relation | -                    |

-   `id`: This field is the primary key, meaning it uniquely identifies each
    record. In a relational database, every table must have a primary key. The
    `index=True` option improves lookup performance when querying on that field.
-   `name`: This field is indexed, meaning it's optimized for searching. The
    `index=True` option improves lookup performance when querying on that field.
-   `email`: This field is unique, ensuring no two accounts have the same email.
-   `balance`: This field has a default value of `0.0`, meaning if no value is
    provided, the account's balance will be set to `0.0`.
-   `transactions`: This is a relationship field which provides access to all
    related `Transaction` records. It doesn't map to a specific column in the
    database, but rather is a collection of `Transaction` instances associated
    with an `Account`. For example, you can access the transactions for a
    specific account using `account.transactions`.

We define the corresponding `Account` model in Python using SQLAlchemy.

```python
class Account(Base):
    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    balance = Column(Float, default=0.0)
    transactions = relationship("Transaction", back_populates="account")
```

### The Primary Key is Automatically Incremented

The `id` field in our `Account` model is automatically managed by SQLAlchemy and
our underlying database system.

When you define a table in SQLAlchemy and specify a primary key field with
`primary_key=True`, the underlying database system is set to automatically
generate a unique value for this field every time a new record is inserted. This
behavior is known as "auto-increment".

In our `Account` model, you've specified
`id = Column(Integer, primary_key=True, index=True)`, which means that `id` is a
primary key and will auto-increment for each new account you add. When you
create a new `Account` instance and add it to the session, SQLAlchemy knows not
to include the `id` in the INSERT statement because it will be automatically
generated by the database.

As a result, when you commit the session after adding an account, the database
generates a unique `id` for that account and SQLAlchemy fetches and assigns that
`id` back to the `id` attribute of the `Account` instance in our Python code.

This is why you don't need to manually add an `id` when creating a new
`Account`. The `id` field is automatically handled by SQLAlchemy and our
database, allowing you to focus on the other fields of our model.

We will see it in action later when we create a new account.

### Transaction Model

The `Transaction` model represents a financial transaction related to an
account. It has five fields: `id`, `account_id`, `amount`, `type`, `timestamp`,
and a `account` relationship field. The schema is represented in table format
below:

| Field      | Type     | Constraint           |
| ---------- | -------- | -------------------- |
| id         | Integer  | Primary Key, Indexed |
| account_id | Integer  | Foreign Key          |
| amount     | Float    | -                    |
| type       | String   | -                    |
| timestamp  | DateTime | -                    |
| account    | Relation | -                    |

-   `id`: This field is the primary key, uniquely identifying each transaction
    record.
-   `account_id`: This field is a foreign key linking the transaction to its
    associated account.
-   `amount`: This field stores the amount involved in the transaction.
-   `type`: This field indicates the type of the transaction (e.g., "deposit",
    "withdrawal", or "transfer").
-   `timestamp`: This field records the date and time when the transaction took
    place.
-   `account`: This is a relationship field that provides access to the
    `Account` associated with the transaction.

We define the corresponding `Transaction` model in Python using SQLAlchemy.

```python
class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id'))
    amount = Column(Float)
    type = Column(String)
    timestamp = Column(DateTime)
    account = relationship("Account", back_populates="transactions")
```

To this end, the `Account` and `Transaction` models define the schema for the
`accounts` and `transactions` tables, respectively, in our relational database.

## Populate the Database

Now that we've set up our database models, we need to populate our database with
data. Populating the database with dummy data allows us to test our application
under more realistic conditions, even before we have actual user data. It helps
us to validate our database schema, relationships, and queries. For this
purpose, we'll use Python's `Faker` library to generate realistic dummy data for
our `accounts` and `transactions` tables.

To automate this process, we have created a script that generates a specified
number of accounts and transactions, then inserts this data into the database.
This script can be run independently or as part of a larger task (like an
application startup script), and is modular enough to be modified or extended as
necessary.

Here is the script that we will use to seed our database:

```python
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
    ...
def create_fake_transaction(account_id: int) -> Transaction:
    ...
def seed_database(num_accounts: int = 50, min_transactions: int = 1, max_transactions: int = 5) -> None:
    ...
if __name__ == "__main__":
    seed_database()
```

This script starts by setting up a connection to our database using SQLAlchemy.
We then define helper functions to create fake `Account` and `Transaction`
instances. The `seed_database` function creates a number of accounts and random
transactions for each account, and then commits these to the database. We wrap
the entire process in a transaction so that we can roll back if any errors
occur. This helps to keep our database in a consistent state even when errors
occur.

To run the script, simply execute the Python file. You can adjust the number of
accounts and transactions created by changing the parameters of the
`seed_database` function call in the `if __name__ == "__main__"` block.

## Querying the Database

Now that we've populated our database with dummy data, we can start querying it:

```python
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
```

## The Engine, Session, and Base

1. `engine = create_engine(SQLALCHEMY_DATABASE_URL)`:

    This line is creating a new instance of SQLAlchemy's engine. The engine is
    the starting point for any SQLAlchemy application. It's "home base" for the
    actual database and its DBAPI, which provides connectivity to a particular
    database server. The SQLAlchemy engine is created by calling the
    `create_engine()` function and passing the connection string of the database
    to connect to. In this case, `SQLALCHEMY_DATABASE_URL` is the connection
    string, which provides the details of the database server (type of database,
    database name, host, user, password, etc.).

2. `SessionLocal = sessionmaker(bind=engine)`:

    This line is creating a factory for SQLAlchemy sessions, bound to our
    engine. A session in SQLAlchemy is a workspace for all operations with the
    database. Objects are added to that session, and they are tracked as they
    are modified. Any changes made won't be persisted into the database until
    the session is committed. The session maker `SessionLocal` is used to get
    new sessions when needed.

3. `Base.metadata.create_all(bind=engine)`:

    This line is issuing a command to the database to create all tables defined
    in our SQLAlchemy models. `Base.metadata.create_all()` goes through each
    SQLAlchemy model, and for each table that doesn't exist in the database, it
    generates the SQL command to create that table. The `bind=engine` parameter
    is telling it where to create these tables. In this case, it's creating the
    tables in the database which the engine is connected to. It's an idempotent
    operation, meaning you can call it multiple times and it will only create
    tables that don't already exist.

To summarize, these lines are setting up the connection to our database and
creating the necessary tables based on our SQLAlchemy models. They're necessary
for our application to interact with the database using SQLAlchemy.

## What is relationship defined in SQLAlchemy?

We have seen earlier in both `Account` and `Transaction` models, there is a
`relationship` attribute. What does it mean?

The `transactions` relationship in the `Account` model is essentially a
convenience for you to access all transactions associated with a specific
account directly from the account instance. This feature is especially powerful
when dealing with relationships (foreign keys) between tables, as it simplifies
the process of querying and navigating these relations.

Let's take an example:

Imagine you've got an instance of `Account` representing a particular bank
account and you'd like to know all the transactions that have occurred with this
account. Without the `transactions` relationship, you'd have to perform a
separate query on the `Transaction` table to get this information:

```python
session = SessionLocal()

ID = 1
account = session.query(Account).filter(Account.id == ID).one()
transactions = session.query(Transaction).filter(Transaction.account_id == account.id).all()
```

Printing out the above would yield

```python
{
    "_sa_instance_state": <sqlalchemy.orm.state.InstanceState object at 0x106153820>,
    "name": "Lindsey Riley",
    "email": "nicholas55@example.com",
    "balance": 3557.707193831535,
    "id": 1,
}
```

and

```python
{
    "_sa_instance_state": <sqlalchemy.orm.state.InstanceState object at 0x106237f40>,
    "amount": 383.6977248919248,
    "account_id": 1,
    "timestamp": datetime.datetime(2023, 3, 5, 8, 5, 55),
    "type": "deposit",
    "id": 1,
}
```

respectively.

However, with the `transactions` relationship established in the `Account`
model, SQLAlchemy automatically retrieves this data for you, and you can access
it like a regular attribute:

```python
account = session.query(Account).filter(Account.id == ID).one()
transactions = account.transactions
```

Printing out the transactions will be exactly the same as how you did it
earlier.

The `relationship` attribute provides a high-level, Pythonic way to retrieve,
add, and remove related objects, which makes it easier to write queries and
manipulate data. It's a powerful feature of SQLAlchemy's ORM that lets you work
with our data in a more intuitive and less error-prone way.

## Session vs. SessionLocal

The terms `SessionLocal` and `Session` refer to different concepts in
SQLAlchemy:

1. `Session`: In SQLAlchemy, a `Session` is the primary interface for
   persistence operations. It's a class that allows you to create instances that
   represent a "workspace" of operations against the database. You use a
   `Session` instance to query the database and persist changes.

2. `SessionLocal`: In the context of FastAPI and SQLAlchemy, `SessionLocal` is
   often used to denote a factory function that produces SQLAlchemy `Session`
   instances. It's typically created by calling `sessionmaker()` with certain
   arguments, such as the database engine or connection, and other configuration
   options.

Here's an example:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("sqlite:///example.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

In this case, `SessionLocal` is a function that, when called, returns a new
SQLAlchemy `Session`. This is useful in FastAPI applications, where you want to
create a new `Session` for each request.

Then, in a FastAPI route, you might use `SessionLocal` to create a new
`Session`:

```python
@app.get("/items/")
def read_items(db: Session = Depends(get_db)):
    items = db.query(Item).all()
    return items
```

In this case, `db` is an instance of `Session` created by calling
`SessionLocal()`. This `Session` is used to query the database and is
automatically closed at the end of the request.

## Object-Relational Mapping (ORM)

### Definition and Intuition

Object-Relational Mapping, or ORM, is a programming technique for converting
data between incompatible type systems in object-oriented programming languages.
In simple terms, it's a way to create, retrieve, update, and delete records from
a relational database using higher-level object-oriented programming languages,
not SQL.

An ORM acts as a translator, allowing you to work with databases using our
programming language of choice instead of writing SQL. It does this by mapping
or tying database tables to classes in a programming language. The rows in these
tables are then tied to instances (objects) of these classes, and columns in the
rows are tied to attributes of these instances.

### Analogy

Let's imagine you speak English, and you're trying to communicate with someone
who speaks French. You don't know any French, and they don't know any English.
You could try to learn French, but that might take a lot of time and effort.
Instead, you hire a translator who knows both languages. You speak in English,
the translator translates our words into French for the other person, and vice
versa. In this analogy, you are the object-oriented programming language, the
French-speaking person is the SQL database, and the translator is the ORM.

### Example

Using Python's SQLAlchemy ORM as an example, let's look at the previously
defined `Account` model:

```python
class Account(Base):
    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    balance = Column(Float, default=0)
    transactions = relationship("Transaction", back_populates="account")
```

Here, the `Account` class is tied to the `accounts` table in the database via
the ORM. Each instance of `Account` corresponds to a row in the `accounts`
table, and each attribute of `Account` corresponds to a column in that row.

With an ORM like SQLAlchemy, you can perform database operations using Python,
and SQLAlchemy translates these operations into SQL for you. For example, to
insert a new account into the database, you could do:

```python
session = SessionLocal()

new_account = Account(name="John Doe", email="john@example.com", balance=100.0)
session.add(new_account)
session.commit()

account_john = session.query(Account).filter(Account.name == "John Doe").one()
pprint(account_john.__dict__)
```

In the background, SQLAlchemy translates this into SQL and executes the SQL
commands for you:

```sql
INSERT INTO accounts (name, email, balance) VALUES ('John Doe', 'john@example.com', 100.0);
```

As you can see, the benefit of using an ORM is that you can work with databases
using the same programming language you're using for the rest of our
application, rather than switching between that language and SQL.

### The Role of Base in SQLAlchemy's ORM System

Object-Relational Mapping (ORM) is a powerful technique that connects the
objects in our code to the tables in our database. In the world of Python, one
of the popular ORMs is SQLAlchemy, and the cornerstone of this system is the
`Base` class.

```python title="app/models/__init__.py" linenums="1"
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
```

When we call `declarative_base()`, we're creating a new base class from which
all of our data models will inherit. This new class, `Base`, is equipped with a
meta-database system that SQLAlchemy uses to orchestrate the communication
between our Python classes and the corresponding database tables.

```python
class Account(Base):
    ...
```

In the example above, `Account` is defined as a subclass of `Base`. By doing so,
`Account` is now recognized by SQLAlchemy as a model that corresponds to a
database table. In practice, this means that SQLAlchemy can now perform a series
of database operations on the `Account` class. These operations include creating
new records, querying existing ones, or updating and deleting records, all using
Python rather than SQL.

When `Account` inherits from `Base`, it's not just inheriting an empty class.
Instead, it's receiving SQLAlchemy's special instrumentation system. This system
includes various methods and attributes that tie `Account` into SQLAlchemy's API
for tasks such as querying, session tracking, and transaction handling.

However, the `Base` class is not a standalone feature. While it does transform
`Account` into an ORM-aware class, its full potential is realized only when used
in conjunction with other tools provided by SQLAlchemy, such as `sessionmaker`
and `create_engine`. These tools work together with `Base` to form a complete
ORM system, facilitating efficient and convenient interaction with our database.

To summarize, SQLAlchemy's `Base` class is a vital piece in the ORM puzzle. It
serves as a translator between our Python classes and the underlying database
tables, simplifying data handling tasks and making our code more maintainable
and scalable. By leveraging the power of `Base` and other SQLAlchemy components,
you can focus more on the core logic of our application and less on the
intricacies of database management.

---

**2. Defining `__tablename__`**

In SQLAlchemy, `__tablename__` is a special attribute you set in our model
classes that inherit from `Base`. It defines the name of the database table the
model class corresponds to.

For example, in the `Account` class, `__tablename__ = "accounts"` tells
SQLAlchemy that the `Account` class corresponds to the `accounts` table in the
database. SQLAlchemy will look for this table when performing operations related
to the `Account` class.

## Moving on from Data (Models) to RESTful API (Endpoints)

As the goal of the article is to guide the reader through building a FastAPI
application, it would be logical to start with setting up the database models
using SQLAlchemy. This foundational step is crucial before we move on to
defining the FastAPI application and its endpoints (requests and responses).

We have explained on how to define the `Account` and `Transaction` models, how
to set up the SQLite engine, and how to create a session for interacting with
the database.

Once the models and database setup are explained, we can then proceed to the
"Creating the FastAPI Application" section. This part would cover how to define
the routes (endpoints) for our API, how to handle incoming requests, and what
responses to return. The readers would learn how to implement all the required
operations (e.g., creating an account, depositing money, etc.) and how to use
the HTTP methods (GET, POST, PUT, DELETE) to manipulate data.

## Endpoints

The very first thing users should understand about RESTful APIs, especially when
dealing with web applications, is the concept of endpoints and HTTP methods. An
**endpoint** refers to a specific URL where an API can be accessed, and an HTTP
method defines what type of operation to perform.

In the context of RESTful APIs, an **endpoint** is a specific URL where our API
can be accessed. Each endpoint is associated with a specific function or
resource. For example, in a banking application, you might have `/accounts`
endpoint for managing accounts and a `/transactions` endpoint for managing
transactions.

Let's say you're creating a RESTful API for a simple banking application. Here
are two basic endpoints: one for managing accounts (`/accounts`), and one for
managing transactions (`/transactions`).

First, you need to import the necessary modules and create an instance of
`FastAPI`:

```python
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Union

app = FastAPI()
```

Then we create a variable `accounts` that is a list of dictionaries. Each
dictionary represents an account and contains the account's `id`, `name`,
`email` and `balance`.

```python
# This list of dictionaries will act as our "database"
accounts = [
    {"id": 1, "name": "John Doe", "email": "johndoe@gmail.com", "balance": 100.0},
    {"id": 2, "name": "Jane Doe", "email": "janedoe@gmail.com", "balance": 200.0},
]


def get_account_by_id(account_id: int) -> Dict[str, Union[int, str, float]]:
    """Return the account with the given id."""
    for account in accounts:
        if account["id"] == account_id:
            return account
    return {}  # Return an empty dict if no account was found
```

The function `get_account_by_id` takes in an `account_id` and returns the
account with the given `account_id` if it exists. If no account is found, it
returns an empty dictionary.

Now, let's create a GET **endpoint** for `/accounts` that retrieves a list of
all accounts:

```python
@app.get("/accounts")
async def get_accounts() -> List[Dict[str, Union[int, str, float]]]:
    """Return all accounts."""
    return accounts
```

Similarly, we can do the same for transactions:

```python
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
```

Let's create a GET **endpoint** for `/transactions` that retrieves a list of all
transactions:

```python
@app.get("/transactions")
async def get_transactions() -> List[Dict[str, Union[int, str, float]]]:
    """Return all transactions."""
    return transactions
```

### Running the FastAPI Application

Once you've designed our models, populated our database, and defined our API
endpoints, the next step is to actually run our FastAPI application.

FastAPI applications are run using an ASGI server. ASGI, or Asynchronous Server
Gateway Interface, is a standard interface between ASGI web servers and Python
web applications. It is an evolution of the older WSGI standard that is built to
handle asynchronous operations and WebSockets in addition to standard HTTP
requests.

The most common ASGI server for FastAPI applications is `uvicorn`. Here's how to
install it:

```bash
pip install uvicorn
```

Once `uvicorn` is installed, you can run our FastAPI application using the
following command:

```bash
uvicorn <APP_NAME>:app --reload
```

In this command, the first `<APP_NAME>` before the colon is the name of our
Python file (i.e., if our python script that holds the app logic is named
`app.py`), without the `.py` extension. The second `app` after the colon is the
FastAPI instance you created in that file (i.e., `app = FastAPI()`). The
`--reload` flag enables hot reloading, which means the server will automatically
update whenever you make changes to our code. This is very useful during
development, but should not be used in a production environment.

Once the server is running, you can open our browser and visit
`http://localhost:8000` to see our application. FastAPI also automatically
generates interactive API documentation for our application. You can access this
by visiting `http://localhost:8000/docs` in our browser. The documentation
allows you to see all our API routes, and you can even try out requests directly
in our browser.

What `uvicorn` does is it essentially acts as the link between our FastAPI
application and the outside world. It listens for incoming HTTP requests, passes
them to our application, and then sends the responses back to the client. The
ASGI specification allows it to handle multiple requests concurrently, which is
a key feature for modern web applications.

Now, if you run our FastAPI application and navigate to
`http://localhost:8000/accounts` in our web browser, you would see a list of all
accounts. Similarly, navigating to `http://localhost:8000/transactions` would
show a list of all transactions. The `@app.get` decorator tells FastAPI that
when a GET request is made to the `/accounts` or `/transactions` endpoint, it
should call the corresponding function (`read_accounts` or `read_transactions`).

So, when you run our FastAPI application and visit
`http://localhost:8000/accounts` in our browser or send a GET request to this
URL using a tool like `curl` or Postman, you're making a request to the
`/accounts` endpoint.

```bash
curl http://localhost:8000/accounts
```

This sends a GET request to the `/accounts` endpoint, and the server should
respond with the JSON data for the accounts, which in our case will be the
hardcoded list of accounts and will return me exactly what I see in the browser

```bash
[
    {
        "id": 1,
        "name": "John Doe",
        "email": "johndoe@gmail.com",
        "balance": 100.0
    },
    {
        "id": 2,
        "name": "Jane Doe",
        "email": "janedoen@gmail.com",
        "balance": 200.0
    }
]
```

Similarly, you can send a GET request to the `/transactions` endpoint like so:

```bash
curl http://localhost:8000/transactions
```

And the server should respond with the hardcoded list of transactions:

```json
[
    {
        "id": 1,
        "account_id": 1,
        "amount": 50.0,
        "type": "deposit",
        "timestamp": "2023-08-05T14:00:00Z"
    },
    {
        "id": 2,
        "account_id": 2,
        "amount": -20.0,
        "type": "withdrawal",
        "timestamp": "2023-08-05T14:00:00Z"
    }
]
```

This way you are communicating with our FastAPI application through its
endpoints using `curl`.

In addition to the URL, each endpoint also involves an HTTP method, which
determines what type of operation is being performed. The most common methods
are GET (retrieve data), POST (send data), PUT (update data), DELETE (remove
data), among others.

In the code example I gave you, the `/accounts` endpoint is associated with the
GET method, meaning it's used to retrieve data. If you wanted to create an
endpoint for sending (posting) data, you would use the `@app.post()` decorator
instead.

Notice that the above examples are all hardcoded values and we did not use our
previously defined database. We shall slowly build up the complexity in the
section [piecing it all together](#piecing-it-all-together).

## What's Async?

Using `async` in our `get_accounts` function declaration allows FastAPI to use
asynchronous I/O (input/output) when handling requests to this endpoint.

When a function is declared with `async`, it becomes a coroutine that can be
paused and resumed, allowing it to yield control while waiting for I/O
operations (like querying a database, requesting data from another server,
reading a file, etc.) to complete. This is extremely beneficial in the context
of a web server, where I/O-bound tasks are common.

In our `get_accounts` function, even though the function itself doesn't perform
any I/O operations or contain any `await` expressions, declaring it as `async`
allows FastAPI to handle requests to this endpoint asynchronously. This means
that while one request is being processed, FastAPI can start processing another.

Here is a simplified example:

1. Request A comes in to get accounts. FastAPI starts processing this request by
   calling our `get_accounts` function.
2. While FastAPI is waiting for the `get_accounts` function to complete for
   Request A, Request B comes in.
3. Because `get_accounts` is a coroutine (declared with `async`), FastAPI can
   pause its execution on Request A and start processing Request B. This is
   possible even if `get_accounts` doesn't actually contain any I/O operations
   or `await` expressions.

So, using `async` allows FastAPI to more efficiently handle multiple requests at
once, improving the overall performance of our application.

## Request

### Intuition

-   An
    **[endpoint](https://www.mulesoft.com/resources/api/what-is-an-endpoint)**
    is a specific URL where an API can be accessed. For instance, `/accounts`
    and `/transactions` are examples of endpoints in our application.

-   A **[request](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview)**
    involves several elements, including:
    -   an
        **[HTTP method](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods)**
        (like GET, POST, PUT, or DELETE)
    -   a **URI** (the endpoint being targeted)
    -   **[headers](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers)**
        (optional metadata about the request)
    -   a **[body](https://developer.mozilla.org/en-US/docs/Web/HTTP/Messages)**
        (data being sent with the request)

The **endpoint** is the destination of a request. You send a **request** to a
particular **endpoint** to perform some operation, and the operation you're
performing is determined by the HTTP method. So in the command
`curl http://localhost:8000/accounts`, `http://localhost:8000/accounts` is the
endpoint and the request is a GET request to that endpoint.

-   The **endpoint** is indeed the "where", the address of the resource you want
    to interact with.
-   The **HTTP method** is the "what", the action you want to perform on the
    resource.
-   The **headers** can be thought of as the "extras", providing additional
    instructions or information about the request.
-   The **body** would be better described as the "content" or "details",
    containing the specific data you want to send or manipulate.

> "The endpoint is the 'where', the HTTP method is the 'what', the headers are
> the 'extras', and the body is the 'content'."

An analogy for this could be sending a letter via post:

-   The endpoint is like the address you're sending our letter to.
-   The HTTP method is like the type of mail you're sending (standard, express,
    registered, etc.).
-   The headers are like any special instructions you give to the post office or
    write on the envelope (handle with care, return receipt requested, etc.).
-   The body is like the content of our letter.

### URI

**URI (Uniform Resource Identifier)** is essentially the URL you're sending a
request to. In the context of APIs, the URI often includes the base URL of the
API server, plus any endpoint-specific path. For example, in
`http://localhost:8000/accounts`, `localhost:8000` is the base URL, and
`/accounts` is the specific endpoint.

The below illustration is taken from
[madewithml](https://madewithml.com/courses/mlops/api/).

<pre class="output ai-center-all" style="padding-left: 0rem; padding-right: 0rem; font-weight: 600;"><span style="color:#d63939">https://</span><span style="color:#206bc4">localhost:</span><span style="color: #4299e1">8000</span><span style="color:#2fb344">/models/{modelId}/</span><span style="color:#ae3ec9">?filter=passed</span><span style="color:#f76707">#details</span>
</pre>

<div class="row">
    <div class="col-md-6">
        <div class="md-typeset__table">
            <div class="md-typeset__scrollwrap">
                <table>
                    <thead>
                        <tr>
                            <th align="left">Parts of the URI</th>
                            <th align="left">Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td align="left"><span style="color: #d63939;">scheme</span></td>
                            <td align="left">protocol definition</td>
                        </tr>
                        <tr>
                            <td align="left"><span style="color: #206bc4;">domain</span></td>
                            <td align="left">address of the website</td>
                        </tr>
                        <tr>
                            <td align="left"><span style="color: #4299e1;">port</span></td>
                            <td align="left">endpoint</td>
                        </tr>
                        <tr>
                            <td align="left"><span style="color: #2fb344;">path</span></td>
                            <td align="left">location of the resource</td>
                        </tr>
                        <tr>
                            <td align="left"><span style="color: #ae3ec9;">query string</span></td>
                            <td align="left">parameters to identify resources</td>
                        </tr>
                        <tr>
                            <td align="left"><span style="color: #f76707;">anchor</span></td>
                            <td align="left">location on webpage</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    <div class="col-md-6">
        <div class="md-typeset__table">
            <div class="md-typeset__scrollwrap">
                <table>
                    <thead>
                        <tr>
                            <th align="left">Parts of the path</th>
                            <th align="left">Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td align="left"><code>/models</code></td>
                            <td align="left">collection resource of all <code>models</code></td>
                        </tr>
                        <tr>
                            <td align="left"><code>/models/{modelID}</code></td>
                            <td align="left">single resource from the <code>models</code> collection</td>
                        </tr>
                        <tr>
                            <td align="left"><code>modelId</code></td>
                            <td align="left">path parameters</td>
                        </tr>
                        <tr>
                            <td align="left"><code>filter</code></td>
                            <td align="left">query parameter</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</div>

We will revisit this section later.

### HTTP Methods

This is the type of HTTP request you're making. The most common methods are GET
(retrieve data), POST (send data), PUT (update data), and DELETE (remove data).
In our examples, we're using the GET method to retrieve data from our API.

The four horsemen GET, POST, PUT, and DELETE are the most common HTTP methods
and are often referred to as
[CRUD](https://en.wikipedia.org/wiki/Create,_read,_update_and_delete) operations
(Create, Read, Update, Delete).

#### GET (Read)

The GET method is used to retrieve data from the server. In a RESTful API, a GET
request to a specific endpoint generally retrieves a list or a specific record.
For example, a GET request to `/accounts` might retrieve a list of all accounts,
while a GET request to `/accounts/{account_id}` would retrieve the details of a
specific account.

We have not implemented the GET request to an individual account yet.

```python
@app.get("/accounts/{account_id}")
async def get_account(account_id: int) -> Dict[str, Union[int, str, float]]:
    """Return the account with the given id."""
    account = get_account_by_id(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    return account
```

Now a GET request to `/accounts/1` would return the details of the account
associated with the id 1.

```json
{
    "id": 1,
    "name": "John Doe",
    "email": "johndoe@gmail.com",
    "balance": 100.0
}
```

Revisiting the URI will yield:

```bash
http://127.0.0.1:8000/accounts/1
```

where `accounts` is the "models" and `1` is the `modelId`, an unique identifier.

#### POST (Create)

The POST method in HTTP is used to send data to the server to create a new
resource. In a RESTful API, a POST request to a specific endpoint generally
creates a new record.

Here's an example of how to implement a POST endpoint to create a new account:

```python
@app.post("/accounts")
async def create_account(
    account: Dict[str, Union[str, float]]
) -> Dict[str, Union[int, str, float]]:
    """Create a new account with the given details."""
    # Here we're just adding the account to our list
    # In a real application, you would save the account to our database
    account_id = max(a["id"] for a in accounts) + 1  # Generate a new ID
    account["id"] = account_id
    accounts.append(account)
    return account
```

In this example, we're using the `@app.post` decorator to create a new POST
endpoint at `/accounts`. The `create_account` function takes a parameter
`account` which is a dictionary representing the account to be created.

You can invoke this endpoint by making a POST request to
`http://127.0.0.1:8000/accounts` with a JSON body containing the account
details. Here's how you can do it with the `curl` command:

```bash
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"name": "Alice", "email": "alice@example.com", "balance": 300.0}' \
     http://127.0.0.1:8000/accounts
```

Here's what each part does:

-   `curl`: This is the command-line tool used to send HTTP requests.
-   `-X POST`: This option specifies the HTTP method of the request, which in
    this case is POST.
-   `-H "Content-Type: application/json"`: This option sets the Content-Type
    header of the request to `application/json`, indicating that the body of the
    request is formatted as JSON.
-   `-d '{"name": "Alice", "email": "alice@example.com", "balance": 300.0}'`:
    This option specifies the body of the request. The `-d` stands for "data".
    The string following `-d` is a JSON object containing the data for the new
    account.
-   `http://127.0.0.1:8000/accounts`: This is the URL that the request is sent
    to. In this case, it's the local address (`127.0.0.1`, also known as
    `localhost`) and port `8000`, followed by the `/accounts` endpoint.

You should get a response with the details of the created account, including its
automatically assigned ID. Note that in a real application, you would likely
want to perform data validation and error checking to ensure that the provided
account details are valid and complete.

#### PUT (Update)

In a RESTful API, a PUT request is generally used to update a specific record.
For example, a PUT request to `/accounts/{id}` with the necessary data in the
request body might update the details of a specific account.

Let's create a `PUT` route to update an account:

```python
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
```

This is a simple example of how you might implement a `PUT` route to update an
account. A `PUT` request to `/accounts/{account_id}` with a JSON body containing
the new account data will update the specified account.

You can test this route with a `curl` command like this:

```bash
curl -X PUT \
     -H "Content-Type: application/json" \
     -d '{"name": "Alice Smith", "balance": 500.0}' \
     http://127.0.0.1:8000/accounts/1
```

Here's what each part does:

-   `curl`: This is the command-line tool used to send HTTP requests.
-   `-X PUT`: This option specifies the HTTP method of the request, which in
    this case is PUT. PUT is typically used for updating an existing resource.
-   `-H "Content-Type: application/json"`: This option sets the Content-Type
    header of the request to `application/json`, indicating that the body of the
    request is formatted as JSON.
-   `-d '{"name": "Alice Smith", "balance": 500.0}'`: This option specifies the
    body of the request. The `-d` stands for "data". The string following `-d`
    is a JSON object containing the new data for the account. In this case,
    we're updating the account's name and balance.
-   `http://127.0.0.1:8000/accounts/1`: This is the URL that the request is sent
    to. In this case, it's the local address (`127.0.0.1`, also known as
    `localhost`) and port `8000`, followed by the `/accounts/1` endpoint. This
    endpoint corresponds to the account with the ID of 1.

When you run this `curl` command, it sends an HTTP PUT request to our FastAPI
application. The application should then update the account with the ID of 1 to
have the new name and balance specified in the request body. If everything goes
as expected, the server should respond with the updated account data.

#### DELETE (Delete)

In a RESTful API, a DELETE request is generally used to delete a specific
record. For example, a DELETE request to `/accounts/{id}` would delete the
specific account.

Let's create a `DELETE` route to delete an account:

```python
@app.delete("/accounts/{account_id}")
async def delete_account(account_id: int) -> Dict[str, str]:
    """Delete the account with the given id."""
    # Find the account to delete
    for account in accounts:
        if account["id"] == account_id:
            # Remove the account from the list
            accounts.remove(account)
            return {"message": f"Account {account_id} deleted successfully"}
    raise HTTPException(status_code=404, detail="Account not found")
```

This is a simple example of how you might implement a `DELETE` route to delete
an account. A DELETE request to `/accounts/{account_id}` will delete the
specified account.

You can test this route with a `curl` command like this:

```bash
curl -X DELETE http://127.0.0.1:8000/accounts/1
```

Here's what each part does:

-   `curl`: This is the command-line tool used to send HTTP requests.
-   `-X DELETE`: This option specifies the HTTP method of the request, which in
    this case is DELETE. DELETE is typically used for deleting an existing
    resource.
-   `http://127.0.0.1:8000/accounts/1`: This is the URL that the request is sent
    to. In this case, it's the local address (`127.0.0.1`, also known as
    `localhost`) and port `8000`, followed by the `/accounts/1` endpoint. This
    endpoint corresponds to the account with the ID of 1.

When you run this `curl` command, it sends an HTTP DELETE request to our FastAPI
application. The application should then delete the account with the ID of 1. If
everything goes as expected, the server should respond with a message indicating
the account was successfully deleted.

### Headers

These provide metadata about the request or response. For example, headers can
be used to specify the format of the data being sent or received (such as JSON),
set cookies, control caching, authenticate the user, and more.

HTTP headers are an important part of the HTTP protocol. They're used to
transmit additional information about the HTTP request or response. HTTP headers
are represented as key-value pairs that are passed along with the request or
response body.

In the context of the curl commands that we discussed, the headers are specified
using the `-H` flag followed by the header name and its value.

Let's look at the command you provided:

```bash
curl -X GET "http://localhost:8000/accounts" \ # method and URI
     -H "accept: application/json" # header, client expects response in JSON
```

Here, `-H "accept: application/json"` is a header. This specific header,
`accept: application/json`, tells the server that the client (in this case, the
curl command) expects the response data to be in JSON format.

Similarly, in the command:

```bash
curl -X POST "http://localhost:8000/accounts" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"name": "Alice", "email": "alice@example.com", "balance": 300.0}'
```

We have two headers: `accept: application/json` and
`Content-Type: application/json`.

The `accept: application/json` header, as before, tells the server that the
client expects the response data to be in JSON format.

The `Content-Type: application/json` header tells the server that the data being
sent by the client (i.e., the request body) is in JSON format.

These headers help ensure that both the client and the server understand the
format of the data being exchanged.

This is a simplified explanation. In practice, HTTP headers can be used to
convey a wide range of information, such as user agent information,
authentication tokens, instructions for caching, and much more.

#### Authorization/Authentication

If you're implementing an authentication system, you'll likely need to work with
headers. For example, a common pattern is to use the Authorization header to
transmit bearer tokens.

### Body

For some types of requests (like POST and PUT), you need to send additional data
along with the request. This data goes in the request body. For example, if
you're creating a new account via a POST request, the details of the account
(like the account name and initial balance) would be sent in the request body.

For example, consider the following curl command:

```bash
curl -X POST "http://localhost:8000/accounts" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"name": "Alice", "email": "alice@example.com", "balance": 300.0}'
```

In this case, the `-d` option is used to specify the body of the request. The
data is a JSON object:

```json
{
    "name": "Alice",
    "email": "alice@example.com",
    "balance": 300.0
}
```

This JSON object contains the data for the new account to be created. The server
will read this data from the body of the request and use it to create the new
account.

The `Content-Type: application/json` header tells the server that the request
body is formatted as JSON.

In the context of FastAPI, the body of the request is typically represented as a
Pydantic model. FastAPI will automatically parse the request body as JSON and
validate it against the Pydantic model. This provides automatic request
validation and serialization.

> Intuitively, body is nothing but the content after `-d` which for instance, if
> I want to create a new account, is just the json content specified after `-d`.

## Response

After the server processes the request, it sends a response back to the client.
This response contains the following components:

-   **Status Line:** This includes the HTTP version, status code, and a short
    message explaining the status code.
-   **Headers:** These provide metadata about the response, such as the content
    type of the response body, set cookies, and more.
-   **Body:** This is the actual data returned by the server. It could contain
    the data requested by a GET request, a confirmation of successful processing
    for a POST or PUT request, error details in case of a failure, or it could
    be empty.

The response body is where you'll find the results of our request. For instance,
if you're fetching data from an API, the response body will contain the
requested data. If you're posting data to an API, the response body will usually
contain a status message or the ID of the created resource.

For instance:

```bash
curl -X GET http://localhost:8000/accounts/1
```

will return:

```json
{
    "id": 1,
    "name": "John Doe",
    "email": "johndoe@gmail.com",
    "balance": 100.0
}
```

and this is the **body response** returned by the server.

Just like requests, responses also have headers. Response headers provide
metadata about the response itself. They can include fields like `Content-Type`
(the format of the response body), `Content-Length` (the length of the response
body in bytes), and `Set-Cookie` (to send cookies from the server to the
client).

If you want to see the response headers as well as the response status, you can
use the `-i` or `--include` option:

```bash
curl --include -X GET http://localhost:8000/accounts/1
```

which outputs

```text
HTTP/1.1 200 OK
date: Fri, 04 Aug 2023 10:03:41 GMT
server: uvicorn
content-length: 70
content-type: application/json

{"id":1,"name":"John Doe","email":"johndoe@gmail.com","balance":100.0}%
```

A key part of the response is the status code. This is a three-digit code that
indicates the result of the request. Here are some common status codes:

| Status Code | Description                                                                                                                           |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 200         | OK - The request was successful.                                                                                                      |
| 201         | Created - The request was successful and a resource was created as a result. This is typically the response sent after POST requests. |
| 204         | No Content - The request was successful, but there is no representation to return (i.e. the response is empty).                       |
| 400         | Bad Request - The request could not be understood or was missing required parameters.                                                 |
| 401         | Unauthorized - Authentication failed or was not provided.                                                                             |
| 403         | Forbidden - Authentication succeeded but the authenticated user does not have access to the requested resource.                       |
| 404         | Not Found - The requested resource could not be found.                                                                                |
| 500         | Internal Server Error - An error occurred on the server.                                                                              |

The response from the server is crucial for understanding the status of our
request. It tells you whether our request was successful and provides the
requested data or helpful error messages if something went wrong. By reading the
response status, headers, and body, you can understand the result of our request
and take appropriate action in our application.

## Filter

Recall the URI we have seen in URI section:

<pre class="output ai-center-all" style="padding-left: 0rem; padding-right: 0rem; font-weight: 600;"><span style="color:#d63939">https://</span><span style="color:#206bc4">localhost:</span><span style="color: #4299e1">8000</span><span style="color:#2fb344">/models/{modelId}/</span><span style="color:#ae3ec9">?filter=passed</span><span style="color:#f76707">#details</span>
</pre>

We have seen up until `{modelId}` but have no clue what is the `?filter=passed`.

Consider this, let's create a new `GET` route to filter accounts based on their
balance. We'll return only the accounts that have a balance greater than a
specified amount. This amount will be provided as a query parameter in the URL.
In this way it is a way to filter the data.

Here's the new route:

```python
@app.get("/accounts/{account_id}/filter")
async def filter_accounts(
    min_balance: float,
) -> List[Dict[str, Union[int, str, float]]]:
    """Return accounts with balance greater than the specified amount."""
    filtered_accounts = [
        account for account in accounts if account["balance"] > min_balance
    ]
    return filtered_accounts
```

With this route, you can send a `GET` request to
`/accounts/filter?min_balance=200` to get all accounts with a balance greater
than 200.

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/accounts/{account_id}/filter?min_balance=10.0' \
  -H 'accept: application/json'
```

This example demonstrates how you can use query parameters to filter data in a
REST API. Note that the `min_balance` parameter in the function signature is
automatically interpreted as a query parameter by FastAPI because it has a
default value.

One warning is that the route must be after `{account_id}`. If you put filter
after accounts, it will be interpreted as a path parameter instead of a query
parameter and you'll get an error `422 Unprocessable Entity`.

## Model-View-Controller (MVC)

While the code we've written so far works, it's not very organized. All the
routes are in one file, and the logic for each route is mixed in with the
request and response code. This is fine for a small application, but as our
application grows, it will become harder to maintain.

### Understanding the Model-View-Controller (MVC) Framework

The Model-View-Controller (MVC) framework is a design pattern widely used in web
development. It separates the application into three interconnected components,
allowing for efficient code organization and modular development. These
components are:

1. **Model:** This component handles the data logic of the application. It
   interacts with the database or data storage to create, read, update, and
   delete (CRUD) data. The model is oblivious to the user interface. In terms of
   a FastAPI application, this could be the SQLAlchemy models defining the
   structure of our data and handling data storage and retrieval.

2. **View:** The view component presents data to the user in a human-readable
   format. It's all about the presentation and how the data is displayed. In a
   traditional web application, this would be the HTML templates that get
   rendered to the user. In a FastAPI application, this could be considered the
   JSON responses that are returned to the client, built using Pydantic models.

3. **Controller:** The controller acts as an intermediary between the model and
   the view. It processes HTTP requests, triggers the data persistence models,
   and sends a response to the client. In other words, it handles the
   application's business logic. In a FastAPI application, this would be our
   route functions (also called view functions or endpoints) that process
   requests and return responses.

Here's a simple way to understand how these components interact:

-   The **user** interacts with the **view** component via the user interface.
-   The **view** sends the user's actions as requests to the **controller**.
-   The **controller** processes the requests and interacts with the **model**
    to perform CRUD operations.
-   The **model** returns any requested data back to the **controller**.
-   The **controller** passes this data to the **view**.
-   The **view** displays this data to the **user**.

In this way, the MVC framework separates concerns into distinct components,
making it easier to manage complex applications. You can work on individual
components without affecting others, which makes debugging and testing more
straightforward and improves the application's scalability and maintainability.

Now, one way we can restructure our FastAPI application is to use the MVC
framework. We can separate our application into three components: models, views,
and controllers. This will help us organize our code and make it easier to
maintain and scale.

```text
.
├── README.md
├── app.py
├── requirements.txt
└── app
    ├── __init__.py
    ├── main.py (this is where you create our FastAPI app instance)
    ├── models
    │   ├── __init__.py
    │   ├── account.py
    │   └── transaction.py
    ├── schemas
    │   ├── __init__.py
    │   ├── account.py
    │   └── transaction.py
    ├── views (or controllers)
    │   ├── __init__.py
    │   ├── account_views.py
    │   └── transaction_views.py
    └── database
        ├── __init__.py
        ├── database.py (this is where you manage our database session)
        └── seed_database.py
```

Here's what each part does:

-   **models**: This is where you define our SQLAlchemy models (i.e., our data
    structures). These are equivalent to the "models" in MVC or MVT.

-   **schemas**: This is where you define our Pydantic models, which you use for
    request validation and response serialization.

-   **views** (or controllers): This is where you define our routes (i.e., our
    view functions or controllers). These are the "views" in MVC or the
    "controllers" in other patterns. Each file in this directory would contain
    the routes related to a specific resource. For instance, `account_views.py`
    might contain routes like `GET /accounts`, `POST /accounts`, etc.

-   **database**: This is where you manage our database session and perform
    database operations. In MVC, these operations are often included in the
    models, but in many modern web applications, it's common to separate them
    out into their own layer.

The `app.py` file at the root of our project is the entry point for our
application. This is where you import and include the routers from our views and
start our ASGI server (e.g., Uvicorn).

### Database Session Management

When building an application, especially one that relies on persistent storage
like a database, it's crucial to set up a reliable and efficient connection
mechanism. The `database.py` file typically serves as the backbone for this. Its
primary responsibilities are:

1. **Establishing a Connection**: It sets up a way for our application to talk
   to the database.
2. **Managing Sessions**: Databases use sessions to encapsulate sequences of
   operations. The `database.py` file sets up a mechanism to create and manage
   these sessions.

Let's break down its components:

#### 1. The Engine

The engine is the starting point for any SQLAlchemy application. It's the home
base for the actual database and its DBAPI, delivered to the SQLAlchemy
application through a connection pool and a Dialect.

```python
from api.conf.base import SQLALCHEMY_DATABASE_URL
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
```

-   **`create_engine`**: This function initializes the connection to the
    database and sets up a pool of connections. The connections in this pool are
    reused whenever an operation needs to talk to the database.

-   **`SQLALCHEMY_DATABASE_URL`**: It's the connection string that tells
    SQLAlchemy how to connect to our database. Its format varies depending on
    the type of database you're using (Postgres, SQLite, MySQL, etc.).

#### 2. Session Factory

In SQLAlchemy, the Session is the primary interface for persistence operations.
It provides the entry point to acquire a new Session when you wish to operate on
the database.

```python
from sqlalchemy.orm import sessionmaker

SessionLocal = sessionmaker(bind=engine)
```

-   **`sessionmaker`**: This function creates a factory that you'll use to get
    new database sessions. The `bind=engine` argument binds the session factory
    to our engine, meaning sessions created with this factory will use the
    connection pool from our engine.

#### 3. Dependency to get a Session

[FastAPI](https://fastapi.tiangolo.com/tutorial/sql-databases/) provides a very
intuitive way to manage database sessions using dependencies. The `get_db`
function is one such dependency.

```python
from sqlalchemy.orm import Session
from typing import Generator

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

-   This function uses the `SessionLocal` factory we defined earlier to get a
    new session.

-   The `yield` keyword in the function makes it a generator. When this function
    is called, it sets up the database session and then yields it. Once the
    request is done and the response is sent, the code after `yield` runs, which
    in this case, closes the session.

By setting up our `database.py` file in this way, you're ensuring efficient use
of resources, a clear separation of concerns, and a structured way to access and
manage our database connections throughout our application.

#### Singleton Pattern

The `create_engine` and `sessionmaker` functions should generally be called once
during the lifecycle of our application. Here's why:

1. **`create_engine`**: This function initializes a connection pool to our
   database. By calling it once, you can ensure that our application reuses
   these connections, which can significantly improve performance. If you were
   to call `create_engine` multiple times, you might end up creating multiple
   connection pools, which would be wasteful and could lead to unexpected
   behavior.

2. **`sessionmaker`**: This function creates a factory for producing new
   database sessions. Once you've bound it to an engine, you can use the
   resulting factory to create new sessions as needed.

For these reasons, it's common to see these lines of code at the module level
(outside of any function or class) in a dedicated database configuration file.
This ensures they are called once when the module is first imported.

Here's a possible structure:

```text
.
├── main.py  (or app.py)
├── database
│   ├── __init__.py
│   ├── session.py   (This is where you can set up our engine, SessionLocal, etc.)
│   ├── base.py  (Optional: If you're using a Base class for our ORM models)
│   └── models
│       ├── __init__.py
│       ├── account.py
│       └── transaction.py
├── schemas
│   ├── __init__.py
│   ├── account.py
│   └── transaction.py
├── views (or controllers)
│   ├── __init__.py
│   ├── account_views.py
│   └── transaction_views.py
└── ...
```

In `session.py`, you might have:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "your_connection_string_here"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
```

Then, whenever you need a session elsewhere in our code, you would import
`SessionLocal` from this module.

### Models

In the context of software applications, particularly those following the MVC
(Model-View-Controller) or similar design patterns, a model is a representation
of a data structure. Essentially, models are blueprints for creating database
tables. They define what kind of data an application will store, how that data
is accessed, and how it relates to other data.

#### SQLAlchemy Models

SQLAlchemy is an Object Relational Mapping (ORM) library for Python, which means
it allows you to work with relational databases in a more Pythonic way. Instead
of writing SQL queries, you can use Python classes and methods to interact with
our database.

In our provided code, you've defined two SQLAlchemy models: `Account` and
`Transaction`. Let's break down what each part does:

1. **Base Class**:

    ```python
    from api.database.base import Base
    ```

    This `Base` class is a base class for all our models, provided by
    SQLAlchemy. It's a foundational piece that provides ORM capabilities to our
    derived models.

2. **Account Model**:

    ```python
    class Account(Base):
        ...
    ```

    Here, you've defined a model for an account. Each account has an `id`,
    `name`, `email`, and `balance`. The `__tablename__` attribute tells
    SQLAlchemy what the table's name should be in the database.

    The `relationship` function defines how this model relates to others. Here,
    it says that an account can have multiple transactions.

3. **Transaction Model**:

    ```python
    class Transaction(Base):
        ...
    ```

    This model represents a transaction. Each transaction has an `id`,
    `account_id`, `amount`, `type`, and `timestamp`. The `account_id` is a
    foreign key that links each transaction to an account.

    Again, the `relationship` function is used, this time to say that a
    transaction is related to an account.

#### Relationships in SQLAlchemy

The relationships between `Account` and `Transaction` models are defined using
the `relationship` function. This function is powerful and provides a way to
define how different models are related in terms of the database.

In our models:

-   An `Account` has a one-to-many relationship with `Transaction`, meaning one
    account can have multiple transactions. This is represented by:

    ```python
    transactions = relationship("Transaction", back_populates="account")
    ```

-   Conversely, a `Transaction` belongs to one `Account`. This is represented
    by:

    ```python
    account = relationship("Account", back_populates="transactions")
    ```

The `back_populates` argument ensures that changes to one side of the
relationship are propagated to the other side. So, if you access the
`transactions` attribute of an `Account` instance, SQLAlchemy will automatically
query and populate that attribute with the related `Transaction` instances. The
same happens in reverse when you access the `account` attribute of a
`Transaction` instance.

In conclusion, models play a crucial role in applications, especially in web
applications where data handling is a primary concern. They provide a structured
way to define, store, and retrieve data. With tools like SQLAlchemy, working
with models becomes intuitive and closely integrated with the programming
language, reducing the need for manual SQL queries and increasing productivity.

### Views (or Controllers)

In an MVC framework, the "view" is the component that handles the presentation
logic. However, in the context of a FastAPI application (or any API-driven
application), the term "view" might be a bit misleading. This is because APIs
generally don't have a traditional "view" in the way that a typical web
application with HTML templates does. Instead, the "view" in a FastAPI
application could be considered the HTTP responses that the application returns
to the client.

However, in the scope of FastAPI (and other similar frameworks), the term
"controller" might be a better fit for this component. Controllers are
responsible for handling incoming HTTP requests and returning responses. They
contain the business logic of the application. In FastAPI, this translates to
our route functions or endpoints.

Let's break down the various parts of a FastAPI controller.

#### Endpoints (or Route Functions)

In FastAPI, an endpoint (also called a route function) is a Python function that
is decorated with one of the HTTP method decorators (like `@app.get()`,
`@app.post()`, etc.). These functions handle HTTP requests, perform the
necessary business logic, and return HTTP responses.

For example:

```python
from fastapi import APIRouter
from sqlalchemy.orm import Session
from api.database.session import get_db

router = APIRouter()

@router.get("/accounts")
def get_accounts(db: Session = Depends(get_db)):
    accounts = db.query(Account).all()
    return accounts
```

In this example, `get_accounts` is an endpoint that handles HTTP GET requests to
the `/accounts` path. When the client makes a GET request to this path, FastAPI
calls the `get_accounts` function and returns whatever that function returns as
the HTTP response.

The function is using a parameter with a default value
(`db: Session = Depends(get_db)`). This parameter is a dependency. When FastAPI
sees this dependency, it will call the `get_db` function, get a database
session, and pass it as the argument to `get_accounts`.

The `get_accounts` function then uses the database session to query all accounts
and return them.

#### APIRouter

FastAPI's `APIRouter` is a powerful tool that allows you to separate our
endpoints into different modules, each with its own `APIRouter`. This makes our
code more modular and easier to maintain and scale. Each `APIRouter` can even
have its own dependencies, exception handlers, etc.

For example:

```python
from fastapi import APIRouter
from . import account_views, transaction_views

router = APIRouter()

router.include_router(account_views.router, prefix="/accounts", tags=["accounts"])
router.include_router(transaction_views.router, prefix="/transactions", tags=["transactions"])
```

In this example, two different routers are included in the main router: one for
accounts and one for transactions. Each has its own prefix and tags.

The `prefix` parameter adds a prefix to all the routes in the included router.
So, for the account views, all routes will start with `/accounts`.

The `tags` parameter groups all the routes in the included router under the
given tags. This is especially useful for interactive API docs UI like Swagger
UI or ReDoc.

Overall, the views (or controllers) are a crucial component of our FastAPI
application. They handle the request-response cycle and contain the business
logic of our application. FastAPI provides many powerful tools to structure and
organize our controllers, allowing for a clean and maintainable codebase.

#### Response Model and Depends

Both `response_model` and `Depends` are important concepts in FastAPI.

**`response_model`** is an optional parameter that you can pass to route
operations like `@app.get()`, `@app.post()`, etc. It's used for:

-   Output data conversion: The model's data (e.g., a SQLAlchemy model instance)
    will be publicly converted according to the `response_model`. For example,
    if the data contains datetime, it will be converted to a string in ISO
    format. This allows FastAPI to understand how to format the response data.
-   Output data validation: FastAPI will check the output data to make sure it's
    valid according to the model. If the data is not valid, FastAPI will raise
    an error, and you can catch these errors while you're still developing our
    application.
-   API documentation: The model will be used to generate the "Response Model"
    section of our API docs UI.

For example, when you have `response_model=List[Account]`, it means that the
endpoint returns a list of Account instances, and FastAPI will handle converting
those instances to JSON.

**`Depends()`** is a function that you use to express dependencies in FastAPI.
When you declare a parameter in a path operation function to be a
`Depends(get_db)`, FastAPI will:

-   Call the `get_db` function.
-   Take the result of that call and pass it as the parameter's value.

This can be used for dependency injection, where you can provide necessary
resources to a function without the function needing to know where those
resources come from. In this case, `get_db` is a function that returns a
database session, so `Depends(get_db)` is a way of telling FastAPI "this
function depends on having a database session".

FastAPI then knows to call `get_db` before calling our path operation function,
and to pass the database session that `get_db` returns into our function. This
can be very useful for ensuring that every request gets a fresh database
session, and for making sure that the session gets closed when the request is
finished.

One important note about `Depends`: it doesn't just handle dependencies on
functions like `get_db`. You can also use it to declare dependencies on other
path operation functions. For example, if you have one path operation function
that shouldn't run until another one has completed, you can use `Depends()` to
express that.

## References and Further Readings

-   [FastAPI](https://fastapi.tiangolo.com/)
-   [Madewithml](https://madewithml.com/courses/mlops/api/)
