from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from . import Base


class Transaction(Base):
    """Transaction Model."""

    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"))
    amount = Column(Float)
    type = Column(String)
    timestamp = Column(DateTime)
    account = relationship("Account", back_populates="transactions")
