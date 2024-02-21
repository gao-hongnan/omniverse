from sqlalchemy.orm import declarative_base

# https://stackoverflow.com/questions/72954928/type-annotations-for-sqlalchemy-model-declaration
Base = declarative_base()  # TODO: migrate to SQLAlchemy 2.0 for better typing support
