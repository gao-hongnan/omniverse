# Banking System

```bash
(venv) $ pip install --editable '.[serving]'
```

```bash
(examples/serving/restful_api/banking/structured) $ export PYTHONPATH=.
(examples/serving/restful_api/banking/structured) $ python omnixamples/software_engineering/serving/restful_api/banking/structured/api/database/seed_database.py
```

Run FastAPI:

```bash
(examples/serving/restful_api/banking/structured) $ uvicorn omnixamples.software_engineering.serving.restful_api.banking.structured.app:app --reload
```

## Terminology

1. Path is Endpoint or Route.
2. What is an endpoint meaning in GET?
3. Path parameters

```python
from fastapi import FastAPI
from typing import Dict

app = FastAPI()


@app.get("/items/{item_id}")
async def read_item(item_id: int) -> Dict[str, int]:
    return {"item_id": item_id}
```

## Dump

The argument `connect_args={"check_same_thread": False}` is needed only for
SQLite. It's not needed for other databases. cite:
https://fastapi.tiangolo.com/tutorial/sql-databases/

-   why crud? By creating functions that are only dedicated to interacting with
    the database (get a user or an item) independent of your path operation
    function, you can more easily reuse them in multiple parts and also add unit
    tests for them.
-   Should we add base class in schema like fastapi?
-   views/accout.py code changed.
