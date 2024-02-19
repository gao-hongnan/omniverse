from http import HTTPStatus
from typing import Any, Dict

from api.database.session import SessionLocal
from api.views import account, transaction
from fastapi import FastAPI, Request

app = FastAPI()

app.include_router(account.router, prefix="/accounts")
app.include_router(transaction.router, prefix="/transactions")

session = SessionLocal()


@app.get("/", tags=["General"])
def _index(request: Request) -> Dict[str, Any]:
    """
    Perform a health check on the server.

    This function is a simple health check endpoint that can be used to
    verify if the server is running correctly. It returns a dictionary
    with a message indicating the status of the server, the HTTP status
    code, and an empty data dictionary.

    Parameters
    ----------
    request : Request
        The request object that contains all the HTTP request
        information.

    Returns
    -------
    response : Dict[str, Any]
        A dictionary containing:
        - message: A string indicating the status of the server. If the
          server is running correctly, this will be "OK".
        - status-code: An integer representing the HTTP status code. If
          the server is running correctly, this will be 200.
        - data: An empty dictionary. This can be used to include any
          additional data if needed in the future.
    """
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK.value,
        "data": {},
        "method": request.method,
        "url": request.url.path,
    }
    return response
