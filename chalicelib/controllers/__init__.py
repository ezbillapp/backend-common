import functools
import operator
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple, Type

from chalice import BadRequestError, ChaliceViewError, ForbiddenError
from sqlalchemy import Float, Integer, Numeric
from sqlalchemy.exc import DatabaseError
from sqlalchemy.orm import Session

from chalicelib.new.config.infra import envars
from chalicelib.new.config.infra.log import logger as _logger
from chalicelib.schema import engine  # pylint: disable=no-name-in-module
from chalicelib.schema.models.model import Model

DECIMAL_PLACES = 6

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


def is_x2m(model: Type[Model], field: str) -> Any:
    rel = model._sa_class_manager.get(field)  # pylint: disable=protected-access
    return rel and getattr(rel.property, "uselist", None) and rel


def _get_filter_m2o(column, op, value, session):
    if value != "any":
        raise BadRequestError("Only value 'any' is accepted in relations")
    column_rel = tuple(column.property.primaryjoin.left.base_columns)[0]
    column_rel_right = tuple(column.property.primaryjoin.right.base_columns)[0]
    expr = column_rel_right.in_(session.query(column_rel))
    if op == "=":
        return expr
    if op == "!=":
        return ~expr
    raise BadRequestError("Only the operators '=' and '!=' are accepted in relations")


def get_filter(model, raw, session=None):
    key, op, value = raw
    column = getattr(model, key)
    if is_m2o(column):
        return _get_filter_m2o(column, op, value, session)
    if is_x2m(model, key):
        return _get_filter_x2m(column, op, value)

    if op == "in":
        return column.in_(value)
    if op == "not in":
        return ~column.in_(value)
    if value == "null":
        value = None
    real_op = operators[op]
    return real_op(column, value)


def filter_query(model, raw_filters, session):
    return [get_filter(model, raw, session) for raw in raw_filters]


def filter_query_doted(model, query, domain: Domain, session):
    join_models = {}
    filters = []
    for raw in domain:
        key, op, value = raw
        tokens = key.split(".")
        relations, field = tokens[:-1], tokens[-1]
        current_model = model
        for rel in relations:
            attrib = getattr(current_model, rel)
            f = getattr(current_model, rel)
            current_model = get_model_from_relationship(attrib)
            join_models[current_model] = f.property.primaryjoin
        filters.append(get_filter(current_model, (field, op, value), session))
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


@contextmanager
def test_session():
    session = Session(bind=engine)
    yield session
    session.rollback()
    session.close()


_session = None


def _local_session():
    global _session  # pylint: disable=global-statement
    if not _session:
        _session = Session(bind=engine)
    try:
        yield _session
        _session.commit()
    except DatabaseError as e:
        _logger.exception(e)
        _session.rollback()
        raise
    finally:
        _session.close()


@contextmanager
def new_session():
    return _local_session() if envars.LOCAL_INFRA else _real_session()


def _real_session():
    session = Session(bind=engine)
    try:
        yield session
        session.commit()
    except DatabaseError as e:
        _logger.exception(e)
        session.rollback()
        raise
    finally:
        session.close()


def add_session(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        session = kwargs.get("session")
        local_new_session = None
        if not session:
            local_new_session = Session(engine)
            kwargs["session"] = local_new_session
        try:
            res = f(*args, **kwargs)
            if local_new_session:
                local_new_session.commit()
        except DatabaseError:
            _logger.exception("IntegrityError")
            if local_new_session:
                local_new_session.rollback()
            raise
        finally:
            if local_new_session:
                local_new_session.close()
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


def is_super_user(context: Dict[str, Any]):
    return "super_user" in context  # TODO user another technique


def ensure_super_user(context: Dict[str, Any], message="do this"):
    if not is_super_user(context):
        raise ForbiddenError(f"Only super users can {message}")


def scale_to_super_user(context: Dict[str, Any] = None) -> Dict[str, Any]:
    if context is None:
        context = {}
    context["super_user"] = True
    return context


def remove_super_user(context: Dict[str, Any] = None):
    if context is None:
        context = {}
    context.pop("super_user", None)
    return context


def _round(value: float) -> float:
    return round(value, DECIMAL_PLACES)


def disable_if_dev(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        DEV_MODE = bool(envars.DEV_MODE)
        if not DEV_MODE:
            return f(*args, **kwargs)

    return wrapper


class ServiceUnavailableError(ChaliceViewError):
    STATUS_CODE = 503


def utc_now():
    return datetime.utcnow().replace(tzinfo=None)
