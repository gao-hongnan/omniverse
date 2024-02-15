# Banking System

```bash
(venv) $ pip install --editable '.[serving]'
```

Change directory:

```bash
(venv) $ cd omnixamples/serving/restful_api/banking/structured
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
