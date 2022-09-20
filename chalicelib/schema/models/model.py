from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.schema import Table

from chalicelib.new.shared.domain.primitives import Identifier, identifier_default_factory
from chalicelib.new.shared.infra.primitives import IdentifierORM

from .. import meta

Base: Any = declarative_base(metadata=meta)


class BasicModel(Base):
    __abstract__ = True
    __table__: Table

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )
    updated_at = Column(
        DateTime,
        onupdate=datetime.utcnow,
    )


class IdentifiedModel(BasicModel):
    __abstract__ = True
    __table__: Table

    identifier = Column(
        IdentifierORM(),
        primary_key=True,
        default=identifier_default_factory,
    )


class Model(BasicModel):
    """Base model for all the models to be persisted in the database"""

    __abstract__ = True
    __table__: Table

    id = Column(  # TODO remove
        Integer,
        primary_key=True,
    )
    identifier = Column(
        IdentifierORM(),
        index=True,
        unique=True,
        # nullable=False, # TODO make not nullable
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
    identifier = Column(
        IdentifierORM(),
        index=True,
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
