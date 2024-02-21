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
