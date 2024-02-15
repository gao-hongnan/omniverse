import os
import uuid
from pathlib import Path

from pydantic import BaseModel
from rich.pretty import pprint


def generate_uuid() -> str:
    """Generate a UUID.

    Returns
    -------
    str
        UUID1 as a string.
    """
    return str(uuid.uuid1())


RUN_ID = generate_uuid()

# Get the directory of the current script
ROOT_DIR = Path(__file__).parents[2].absolute()

# Construct the path to the database file
DATABASE_URL = ROOT_DIR / "api" / "database" / "database.db"

# Set up database URL as an environment variable for better flexibility
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_URL}")

SEED = 42


# TODO: explore how to use model_post_init to make run_id inside Config
class Config(BaseModel):
    pass


if __name__ == "__main__":
    config = Config()
    pprint(config)
