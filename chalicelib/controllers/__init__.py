import functools
import operator
from typing import Any, Dict, List, Set, Tuple

from chalice import BadRequestError
from sqlalchemy import Float, Integer, Numeric
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..schema import engine

Domain = List[Tuple[str, str, Any]]
SearchResult = List[Dict[str, Any]]
SearchResultPaged = Tuple[SearchResult, int]


operators = {
    "<": operator.lt,
    "<=": operator.le,
    "=": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
    "not": operator.not_,
    "is": operator.is_,
    "is not": operator.is_not,
}


MISSING_FIELD = False
NUMERICS = {Integer, Float, Numeric}


def get_filter(model, raw):
    key, op, value = raw
    column = getattr(model, key)
    if hasattr(column.property, "mapper"):
        if value != "any":
            raise BadRequestError("Only value 'any' is accepted in relations")
        if op == "=":
            return column.any()
        if op == "!=":
            return ~column.any()
        raise BadRequestError("Only the opertators '=' and '!=' are accepted in relations")

    if op == "in":
        return column.in_(value.split(","))
    real_op = operators[op]
    if value == "null":
        value = None
    return real_op(column, value)


def filter_query(model, raw_filters):
    return [get_filter(model, raw) for raw in raw_filters]


def ensure_list(f):
    @functools.wraps(f)
    def wrapper(cls, records, *args, **kwargs):
        if not isinstance(records, List):
            records = [records]
        return f(cls, records, *args, **kwargs)

    return wrapper


def ensure_set(f):
    @functools.wraps(f)
    def wrapper(cls, ids, *args, **kwargs):
        if not isinstance(ids, Set):
            ids = {*ids} if isinstance(ids, list) else {ids}
        return f(cls, ids, *args, **kwargs)

    return wrapper


def add_session(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        session = kwargs.get("session")
        new_session = None
        if not session:
            new_session = Session(engine)
            kwargs["session"] = new_session
        res = f(*args, **kwargs)
        if new_session:
            try:
                new_session.commit()
            except IntegrityError as exception:
                raise BadRequestError("Internal exception in the DB") from exception
            finally:
                new_session.close()
        return res

    return wrapper
