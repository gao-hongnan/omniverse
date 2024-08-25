from sqlalchemy import Column, Float, Integer, String
from sqlalchemy.orm import relationship

from . import Base


class Account(Base):  # type: ignore[misc]
    """Account Model."""

    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    balance = Column(Float, default=0)
    transactions = relationship("Transaction", back_populates="account")
