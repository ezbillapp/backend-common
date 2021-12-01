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


def get_model_from_relationship(relationship):
    return relationship.property.mapper.class_


def _get_x2m_cardinal_filter(column, op, value):
    if value != "any":
        raise BadRequestError("Only value 'any' is accepted in relations")
    if op == "=":
        return column.any()
    if op == "!=":
        return ~column.any()


def _get_x2m_relational_filter(column, op, value):
    if not isinstance(value, list):
        raise BadRequestError(f"Invalid value for m2m field {column}, must be a list")
    if op not in ("in", "not in"):
        raise BadRequestError(f"Invalid operator {op} for m2m field")
    rel_model = get_model_from_relationship(column)
    res = column.any(rel_model.id.in_(value))
    if op == "not in":
        res = ~res
    return res


def _get_filter_x2m(column, op, value):
    if op in ("=", "!="):
        return _get_x2m_cardinal_filter(column, op, value)
    return _get_x2m_relational_filter(column, op, value)


def is_m2o(column):
    return hasattr(column.property, "mapper") and not hasattr(column.property, "uselist")


def is_x2m(model, field: str) -> bool:
    rel = model._sa_class_manager.get(field)  # pylint: disable=protected-access
    return rel and getattr(rel.property, "uselist", None) and rel


def _get_filter_m2o(column, op, value):
    if value != "any":
        raise BadRequestError("Only value 'any' is accepted in relations")
    if op == "=":
        return column.has()
    if op == "!=":
        return ~column.has()
    raise BadRequestError("Only the opertators '=' and '!=' are accepted in relations")


def get_filter(model, raw):
    key, op, value = raw
    column = getattr(model, key)
    if is_m2o(column):
        return _get_filter_m2o(column, op, value)
    if is_x2m(model, key):
        return _get_filter_x2m(column, op, value)

    if op == "in":
        return column.in_(value)
    real_op = operators[op]
    if value == "null":
        value = None
    return real_op(column, value)


def filter_query(model, raw_filters):
    return [get_filter(model, raw) for raw in raw_filters]


def ensure_list(f):
    @functools.wraps(f)
    def wrapper(cls, records, *args, **kwargs):
        if records is None:
            records = []
        elif not isinstance(records, List):
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
