import functools
from datetime import datetime
from typing import Any, Callable, Dict, List, Set, Tuple, Type, Union

import unidecode
from chalice import ForbiddenError, NotFoundError, UnauthorizedError
from chalice.app import MethodNotAllowedError
from sqlalchemy import or_, text
from sqlalchemy.orm import relationship
from sqlalchemy.sql.functions import ReturnTypeFromArgs

from ..schema.models import Company, Model, Permission, User, Workspace
from . import (
    Domain,
    SearchResult,
    SearchResultPaged,
    add_session,
    ensure_list,
    ensure_set,
    filter_query,
)


class unaccent(ReturnTypeFromArgs):  # pylint: disable=too-many-ancestors
    pass


def check_context(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        context = kwargs.get("context", {}) or {}
        kwargs["context"] = context
        res = f(*args, **kwargs)
        if not context.get("super_user") and not context.get("guest_user"):  # TODO
            cls = args[0]
            session = kwargs["session"]
            cls.check_companies(res, session=session, context=context)
        return res

    return wrapper


class CommonController:
    model: Type[Model]
    BASE_FIELDS = {
        "id",
        "created_at",
    }

    restricted_fields = {
        "id",
        "created_at",
        "updated_at",
    }
    restricted_update_fields = {
        "id",
        "created_at",
        "updated_at",
    }
    default_read_fields = {
        "id",
    }

    pseudo_enums: Dict[str, Set[str]] = {}

    fuzzy_fields: Tuple[Any, ...] = ()
    fuzzy_has: Dict[relationship, Tuple[str, ...]] = {}
    _order_by: str = ""

    @classmethod
    def get_controllers_by_model(cls) -> Dict[Model, "CommonController"]:
        controllers = CommonController.__subclasses__()
        return {controller.model: controller for controller in controllers}

    @classmethod
    @add_session
    def ensure_role_access(cls, records: List[Model], *, session=None, context=None):
        """To implement in each model"""

    @classmethod
    @add_session
    def _fuzzy_search(cls, query, fuzzy_search, *, session=None):
        # REATE EXTENSION unaccent;
        fuzzy_search = unidecode.unidecode(fuzzy_search)
        fuzzy_filter = tuple(
            unaccent(field).ilike(f"%{fuzzy_search}%") for field in cls.fuzzy_fields
        )
        fuzzy_sub_filter = tuple(
            field.has(unaccent(subfield).ilike(f"%{fuzzy_search}%"))
            for field, subfields in cls.fuzzy_has.items()
            for subfield in subfields
        )
        query = query.filter(
            or_(
                *fuzzy_filter,
                *fuzzy_sub_filter,
            )
        )
        return query

    @classmethod
    @add_session
    def _get_default_order_by(cls, *, session=None):
        order_by = ""
        if cls._order_by:
            order_by = cls._order_by
        elif "name" in cls.model.__table__.c:
            order_by = "name"
        return order_by + ", id" if order_by else "id"

    @classmethod
    @add_session
    @check_context
    def _search(
        cls,
        domain: Domain,
        order_by: str = "",
        limit: int = None,
        offset: int = 0,
        active: bool = True,
        fuzzy_search: str = None,
        *,
        session=None,
        context=None,
    ) -> List[Model]:
        domain_parsed = filter_query(cls.model, domain)
        query = session.query(cls.model)
        if fuzzy_search:
            query = cls._fuzzy_search(query, fuzzy_search, session=session)
        query = query.filter(*domain_parsed)
        if "active" in cls.model.__table__.c:
            query = query.filter(cls.model.active == active)
        if not order_by:
            order_by = cls._get_default_order_by(session=session)
        query = query.order_by(f"{text(order_by)}")
        query = query.limit(limit + 1) if limit else query
        query = query.offset(limit * offset if (limit and offset) else 0)
        records = query.all()
        cls.ensure_role_access(records, session=session, context=context)
        return records

    @classmethod
    @add_session
    def search(
        cls,
        domain: Domain,
        order_by: str,
        limit: int,
        offset: int = 0,
        active: bool = True,
        fuzzy_search: str = None,
        *,
        session=None,
        context=None,
    ) -> SearchResultPaged:
        next_page = False
        records = cls._search(
            domain,
            order_by,
            limit,
            offset,
            active,
            fuzzy_search,
            session=session,
            context=context,
        )
        if limit and len(records) > limit:
            records.pop()
            next_page = True
        return records, next_page

    @classmethod
    @add_session
    def all_ids(cls, *, session=None) -> Set[int]:
        records = session.query(cls.model.id).all()
        return {record[0] for record in records}

    @classmethod
    def _check_data_key_value(cls, key, value):
        if key in cls.restricted_fields:
            raise ForbiddenError("The field '{}' can not be setted manually".format(key))
        if key in cls.pseudo_enums and value not in cls.pseudo_enums[key]:
            raise ForbiddenError("The field '{}' don not supports '{}' value".format(key, value))

    @classmethod
    @add_session
    def _check_to_update_data(cls, data: Dict[str, Any], *, session=None, context=None):
        for key in data:
            if key in cls.restricted_update_fields:
                raise ForbiddenError("The field '{}' can not be updated manually".format(key))

    @classmethod
    @add_session
    def _check_data(cls, record: Model, data: Dict[str, Any], *, session=None, context=None):
        is_active = getattr(record, "active", None)
        if is_active is None:
            is_active = True
        if not is_active and not data.get("active", False):
            raise ForbiddenError(
                f"The inactive records {cls.log_records(record)} cannot be updated"
            )
        if context.get("super_user"):
            return
        for key, value in data.items():
            cls._check_data_key_value(key, value)

    @classmethod
    @add_session
    @check_context
    def create(cls, data: Dict[str, Any], *, session=None, context=None):
        try:
            m2m = []
            for key, value in data.copy().items():
                m2m_rel = cls.is_m2m(cls.model, key)
                if m2m_rel:
                    data.pop(key)
                    m2m.append((m2m_rel, key, value))
                    continue
            record = cls.model(**data)
            for rel, key, value in m2m:
                field = getattr(record, key)
                cls._set_m2m(rel.property.entity, field, value, session=session)
        except TypeError as e:
            raise ForbiddenError(e) from e
        cls._check_data(record, data, session=session, context=context)
        session.add(record)
        return record

    @classmethod
    def is_m2m(cls, record, field: str) -> bool:
        rel = record._sa_class_manager.get(field)  # pylint: disable=protected-access
        return rel and getattr(rel.property, "uselist", None) and rel

    @classmethod
    def record_to_dict(cls, record: Model, fields_string: Set["str"]) -> Dict[str, Any]:
        """Return a dictionary with the fields given (can use dot to indicate subfields) filled with the object data

        Args:
            record (Model): Record to convert
            fields_string (Set[): Fields to retrieve

        Returns:
            Dict[str, Any]: Dictionary with the fields filled with the object data
        """

        def tokenize(fields: Set[str]) -> Dict[str, Any]:
            """Split the fields using the dot character, creating a dict"""
            result = {}
            for field in fields:
                parts = field.split(".")
                current = result
                for part in parts:
                    current = current.setdefault(part, {})
            return result

        converters = {
            datetime: lambda x: x.isoformat(),
        }

        controllers_by_model = cls.get_controllers_by_model()

        def obj_to_dict(obj, tokens: Dict[str, Any]):
            """Create a dictionary based on the token fields"""
            if obj is None:
                return None
            result = {}
            for key, value in tokens.items():
                real_value = getattr(obj, key)
                m2m = cls.is_m2m(obj, key)
                if value == {} and not isinstance(real_value, Model) and not m2m:
                    if real_value.__class__ in converters:
                        real_value = converters[real_value.__class__](real_value)
                    result[key] = real_value
                else:
                    controller = controllers_by_model.get(real_value.__class__, cls)
                    value.update({f: {} for f in controller.default_read_fields})
                    if m2m or isinstance(real_value, list):
                        result[key] = [obj_to_dict(x, value) for x in real_value]
                    else:
                        result[key] = obj_to_dict(getattr(obj, key), value)
            return result

        tokens = tokenize(fields_string)
        return obj_to_dict(record, tokens)

    @classmethod
    @add_session
    @ensure_list
    def detail(
        cls,
        records: List[Model],
        fields: Set[str] = None,
        *,
        session=None,
        context=None,
    ) -> SearchResult:
        fields = set(fields or []) | cls.default_read_fields
        session.add_all(records)
        cls.ensure_role_access(records, session=session, context=context)
        return [cls.record_to_dict(record, fields) for record in records]

    @classmethod
    @add_session
    def get_user_companies(cls, user: User, *, session=None) -> Set[Company]:
        session.add(user)
        permissions = session.query(Permission).filter(Permission.user_id == user.id).all()
        return {permission.company for permission in permissions}

    @classmethod
    @add_session
    @ensure_list
    def check_companies(cls, records: List[Model], *, session=None, context=None):
        user = context["user"]
        session.add(user)
        allowed_companies = cls.get_user_companies(user, session=session)
        allowed_companies_ids = {company.id for company in allowed_companies}
        session.add_all(records)
        company_ids = cls.get_company_ids(records, session=session)
        not_allowed_companies = company_ids - allowed_companies_ids
        if not_allowed_companies:
            raise UnauthorizedError(f"Companies `{not_allowed_companies}` not allowed")

    @classmethod
    @ensure_list
    def log_records(cls, records):
        if not records:
            return ""
        model = records[0].__class__.__name__
        return f"{model}({', '.join([str(r.id) for r in records])})"

    @classmethod
    @add_session
    @check_context
    @ensure_set
    def get(cls, ids: Set[int], *, session=None, context=None, singleton=True) -> List[Model]:
        records = (
            session.query(cls.model).filter(cls.model.id.in_(ids)).order_by(cls.model.id).all()
        )
        if len(ids) != len(records):
            ids_readed = {record.id for record in records}
            diff = ids - ids_readed
            raise NotFoundError(f"ID's: {diff} were not found in model {cls.model.__name__}")
        cls.ensure_role_access(records, session=session, context=context)
        return records[0] if len(ids) == 1 and singleton else records

    @classmethod
    @add_session
    def get_company_ids(cls, records: List[Model], *, session=None) -> Set[int]:
        session.add_all(records)
        company_ids = set()
        for record in records:
            if record.__table__.c.get("company_id") is not None:
                company_ids.add(record.company_id)
            else:
                raise UnauthorizedError(
                    f"The model {cls.model.__name__} have not function to get companies"
                )
        return company_ids

    @classmethod
    @add_session
    def _remove_from_m2m(cls, model, field, records: int, *, session=None):
        for record in records:
            if record in field:
                field.remove(record)
            else:
                raise NotFoundError(cls.log_records(record))

    @classmethod
    @add_session
    def _delete_m2m_rel(cls, model, field, records: int, *, session=None):
        for record in records:
            if record in field:
                session.delete(record)
            else:
                raise NotFoundError(cls.log_records(record))

    @classmethod
    @add_session
    def _add_to_m2m(cls, model, field, records: int, *, session=None):
        for record in records:
            if record in field:
                raise ForbiddenError(f"{cls.log_records(record)} already in relation")
        field.extend(records)

    @classmethod
    @add_session
    def _remove_all_from_m2m(cls, model, field, rel_id: int, *, session=None):
        field.clear()

    @classmethod
    @add_session
    def _replace_all_from_m2m(cls, model, field, records: int, *, session=None):
        field.clear()
        records = [records] if not isinstance(records, list) else records
        field.extend(records)

    @classmethod
    @add_session
    def _set_m2m(
        cls, model, field, value: List[Tuple[int, Union[int, None, List[int]]]], *, session=None
    ):
        """Update an m2m field based on the next structure:
        (0, None): NotImplemented
        (1, id): NotImplemented
        (2, id): Revemove but NOT delete from the DB
        (3, id): Remove and delete from the DB
        (4, id): Add in relation
        (5, None): Remove all ids from the relation
        (6, ids): Replace all current ids with the providen ids

        Args:
            field ([type]): [description]
            value (List[Tuple[int, Union[int, None, List[int]]]]): [description]
            session ([type], optional): [description]. Defaults to None.
        """
        actions = {
            0: NotImplementedError,
            1: NotImplementedError,
            2: cls._remove_from_m2m,
            3: cls._delete_m2m_rel,
            4: cls._add_to_m2m,
            5: cls._remove_all_from_m2m,
            6: cls._replace_all_from_m2m,
        }
        for action, ids in value:
            if action not in actions or actions[action] is NotImplementedError:
                raise MethodNotAllowedError(f"Action {action} not implemented")
            ids = ids if isinstance(ids, list) else ids and [ids] or []
            records = [session.query(model).get(id) for id in ids]
            if None in records:
                raise NotFoundError(f"ID's: {ids} not found")
            actions[action](model, field, records, session=session)

    @classmethod
    @add_session
    @ensure_list
    @check_context
    def update(
        cls,
        records: List[Model],
        data: Dict[str, Any],
        *,
        session=None,
        context=None,
    ) -> List[Model]:
        session.add_all(records)
        cls.ensure_role_access(records, session=session, context=context)
        for record in records:
            cls._check_data(record, data, session=session, context=context)
            cls._check_to_update_data(data, session=session, context=context)
            for key, value in data.items():
                m2m_rel = cls.is_m2m(record, key)
                if m2m_rel:
                    field = getattr(record, key)
                    cls._set_m2m(m2m_rel.property.entity, field, value, session=session)
                    continue
                setattr(record, key, value)
            record.updated_at = datetime.now()
        return records

    @classmethod
    @add_session
    @ensure_list
    def delete(cls, records: List[Model], *, session=None, context=None) -> Set[int]:
        session.add_all(records)
        cls.check_companies(records, session=session, context=context)
        cls.ensure_role_access(records, session=session, context=context)
        ids = [record.id for record in records]
        session.query(cls.model).filter(cls.model.id.in_(ids)).delete()
        return set(ids)

    @classmethod
    @add_session
    @ensure_list
    @check_context
    def toggle_archive(cls, records: List[Model], *, session=None, context=None) -> List[Model]:
        session.add_all(records)
        base_status = records[0].active
        cls.ensure_role_access(records, session=session, context=context)
        if any(base_status != record.active for record in records):
            raise ForbiddenError("All the records must been in the same status")
        new_status = not base_status
        cls.update(records, {"active": new_status}, session=session, context=context)
        return records

    @classmethod
    @add_session
    def get_owned_by(cls, user: User, *, session=None, context=None) -> List[Company]:
        session.add(user)
        return session.query(Workspace).filter(Workspace.owner_id == user.id).all()


def get_m2m_repr(m2m_rel, attribs: Set[str]) -> List[Dict[str, Any]]:
    return [
        {
            attrib: getattr(
                rel,
                attrib,
            )
            for attrib in attribs
        }
        for rel in m2m_rel
    ]


def add_if_exists(
    data: Dict[str, Any],
    record: Model,
    main_attrib: str,
    sub_attrib: str,
    mapper: Callable[[Any], Any] = None,
):
    if mapper is None:
        mapper = lambda x: x
    record_attrib = getattr(record, main_attrib)
    data[main_attrib] = (
        {
            **data.get(main_attrib, {}),
            "id": record_attrib.id,
            sub_attrib: mapper(getattr(record_attrib, sub_attrib)),
        }
        if record_attrib
        else None
    )
