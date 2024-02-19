# Banking System

```bash
(venv) $ pip install --editable '.[serving]'
```

Change directory:

```bash
(venv) $ cd omnixamples/software_engineering/serving/restful_api/banking/structured
```

```bash
(examples/serving/restful_api/banking/structured) $ export PYTHONPATH=.
(examples/serving/restful_api/banking/structured) $ python api/data/seed_database.py
```

Run FastAPI:

```bash
(examples/serving/restful_api/banking/structured) $ uvicorn app:app --reload
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
