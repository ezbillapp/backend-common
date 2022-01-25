from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.schema import Table

from .. import meta

Base: Any = declarative_base(metadata=meta)


class Model(Base):
    """Base model for all the models to be peristed in the database"""

    __abstract__ = True
    __table__: Table

    id = Column(
        Integer,
        primary_key=True,
    )
    created_at = Column(
        DateTime,
        default=datetime.now,
    )
    updated_at = Column(
        DateTime,
    )


class CodeName(Base):
    """Base model for all the models to be peristed in the database
    using Code and Name as all fields"""

    __abstract__ = True
    __table__: Table

    id = Column(
        Integer,
        primary_key=True,
    )
    code = Column(
        String,
        index=True,
        nullable=False,
        unique=True,
    )
    name = Column(
        String,
        index=True,
        nullable=False,
    )
