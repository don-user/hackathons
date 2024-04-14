from typing import Optional
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
)
from sqlalchemy.sql import func


from app.store.database.sqlalchemy_base import db


@dataclass
class DocsBase:
    id: Optional[int]
    time_create: datetime
    filename: str
    content: str
    label: str


class DocsModel(db):
    __tablename__ = "docs"
    id = Column(Integer, primary_key=True)
    time_create = Column(
        DateTime(timezone=True), index=True, server_default=func.now(), nullable=False
    )
    filename = Column(String, nullable=False, unique=True)
    content = Column(Text, nullable=False)
    label = Column(String, nullable=False)
