from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.schema import Table

from .. import meta

Base: Any = declarative_base(metadata=meta)


class Model(Base):
    """Base model for all the models to be peristed in the database"""

    __abstract__ = True
    __table__: Table

    id = Column(  # TODO remove
        Integer,
        primary_key=True,
    )
    identifier = Column(
        UUID(as_uuid=True),
        index=True,
        unique=True,
        # nullable=False, # TODO make not nullable
    )
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )
    updated_at = Column(
        DateTime,
        onupdate=datetime.utcnow,
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
