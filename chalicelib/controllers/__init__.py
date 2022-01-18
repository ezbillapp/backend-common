import functools
import operator
from typing import Any, Dict, List, Set, Tuple, Type

from chalice import BadRequestError
from sqlalchemy import Float, Integer, Numeric
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from chalicelib.schema import engine
from chalicelib.schema.models.model import Model

Domain = List[Tuple[str, str, Any]]
SearchResult = List[Dict[str, Any]]
SearchResultPaged = Tuple[SearchResult, bool, int]


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


def get_model_from_relationship(relationship) -> Model:
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
    return hasattr(column.property, "mapper") and not getattr(column.property, "uselist")


def is_x2m(model: Type[Model], field: str) -> bool:
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
    if value == "null":
        value = None
    real_op = operators[op]
    return real_op(column, value)


def filter_query(model, raw_filters):
    return [get_filter(model, raw) for raw in raw_filters]


def filter_query_doted(model, query, domain: Domain):
    join_models = {}
    filters = []
    for raw in domain:
        key, op, value = raw
        tokens = key.split(".")
        rels, field = tokens[:-1], tokens[-1]
        current_model = prev_model = model
        for rel in rels:
            attrib = getattr(current_model, rel)
            tmp_model = current_model
            current_model = get_model_from_relationship(attrib)
            if is_x2m(tmp_model, rel):
                rel_id = f"{prev_model.__table__.name}_id"
                join_models[current_model] = getattr(current_model, rel_id) == prev_model.id
            else:
                rel_id = f"{rel}_id"
                join_models[current_model] = current_model.id == getattr(prev_model, rel_id)
            prev_model = current_model
        real_op = operators[op]
        column = getattr(current_model, field)
        filter = real_op(column, value)  # type: ignore
        filters.append(filter)
    for jm, on in join_models.items():
        query = query.join(jm, on)
    return query.filter(*filters)


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


def ensure_dict_by_ids(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        dict_by_ids = args[1]
        if not isinstance(dict_by_ids, Dict):
            raise BadRequestError("Invalid dict_by_ids")
        dict_by_ids = {int(k): v for k, v in dict_by_ids.items()}
        args = args[:1] + (dict_by_ids,) + args[2:]
        return f(*args, **kwargs)

    return wrapper
