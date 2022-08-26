import csv
import enum
import functools
import io
from datetime import date, datetime
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, List, Set, Tuple, Type, Union
from zipfile import ZipFile

import boto3
import requests
import unidecode
from chalice import ForbiddenError, NotFoundError, UnauthorizedError
from chalice.app import MethodNotAllowedError  # type: ignore
from chalicelib.controllers import (
    Domain,
    SearchResult,
    SearchResultPaged,
    add_session,
    ensure_list,
    ensure_set,
    filter_query,
    filter_query_doted,
    is_super_user,
    is_x2m,
    utc_now,
)
from chalicelib.new.config.infra import envars
from chalicelib.new.config.infra.log import logger as _logger
from chalicelib.new.shared.domain.primitives import Identifier, identifier_default_factory
from chalicelib.schema.models import (  # pylint: disable=no-name-in-module
    Company,
    Model,
    Permission,
    User,
    Workspace,
)
from openpyxl import Workbook  # type: ignore
from sqlalchemy import or_, text
from sqlalchemy.orm import Query, relationship
from sqlalchemy.sql.functions import ReturnTypeFromArgs

EXPORT_EXPIRATION = 60 * 60 * 24 * 7

primitives = {
    str,
    int,
    float,
    bool,
    date,
    datetime,
}
PrimitiveType = Union[str, int, float, bool, date, datetime]


class ExportFormat(enum.Enum):
    CSV = enum.auto()
    PDF = enum.auto()
    XLSX = enum.auto()
    XML = enum.auto()


class unaccent(ReturnTypeFromArgs):  # pylint: disable=too-many-ancestors
    inherit_cache = True


def _plain_field(record: Model, field_str: str) -> Any:
    res = record
    for part in field_str.split("."):
        if not res:
            continue
        res = getattr(res, part)
    return CommonController._to_primitive(res)  # pylint: disable=protected-access


def check_context(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        context = kwargs.get("context", {}) or {}
        kwargs["context"] = context
        res = f(*args, **kwargs)
        if not is_super_user(context) and not context.get("guest_user"):  # TODO
            cls = args[0]
            session = kwargs["session"]
            cls.check_companies(res, session=session, context=context)
        return res

    return wrapper


resume_fields = (
    "Tipo",
    "Conteo de CFDIs",
    "Retención IVA",
    "Retención IEPS",
    "Retención ISR",
    "Traslado IVA",
    "Traslado IEPS",
    "Traslado ISR",
    "Impuesto Local",
    "Subtotal",
    "Descuento",
    "Neto",
    "Total",
)


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
        "identifier",
    }
    onchange_functions: Dict[str, List[Callable]] = {}

    pseudo_enums: Dict[str, Set[str]] = {}

    fuzzy_fields: Tuple[Any, ...] = ()
    fuzzy_has: Dict[relationship, Tuple[str, ...]] = {}
    _order_by: str = ""

    @classmethod
    def get_controllers_by_model(cls) -> Dict[Type[Model], Type["CommonController"]]:
        controllers = CommonController.__subclasses__()
        return {getattr(controller, "model", None): controller for controller in controllers}  # type: ignore

    @classmethod
    @add_session
    def ensure_role_access(cls, records: List[Model], *, session=None, context=None):
        """To implement in each model"""

    @classmethod
    @add_session
    def _fuzzy_search(cls, query, fuzzy_search, *, session=None):
        # CREATE EXTENSION unaccent;
        if not fuzzy_search:
            return query
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
        return order_by or "id"

    @staticmethod
    def _normalize_order_by(model: Type[Model], order_by: str) -> str:
        table_name = model.__table__.name
        order_by = order_by.replace(f"{table_name}.", "").replace('"', "")
        parts = order_by.split(",")
        scaped = []
        for part in parts:
            components = part.strip().split(" ")
            column = components[0]
            order_mode = components[1] if len(components) > 1 else "asc"
            scaped.append(f'{model.__table__.name}."{column}" {order_mode}')
        return ", ".join(scaped)

    @classmethod
    def apply_domain(cls, query, domain: List[Tuple[str, str, Any]], session):
        domain_doted = []
        domain_no_doted = []
        for t in domain:
            if t[0].find(".") != -1:
                domain_doted.append(t)
            else:
                domain_no_doted.append(t)
        query = filter_query_doted(cls.model, query, domain_doted, session)

        domain_parsed = filter_query(cls.model, domain_no_doted, session)
        query = query.filter(*domain_parsed)
        return query

    @classmethod
    @add_session
    def _search(
        cls,
        domain: Domain,
        order_by: str = "",
        limit: int = None,
        offset: int = 0,
        active: bool = True,
        fuzzy_search: str = None,
        *,
        need_count: bool = False,
        session=None,
        context=None,
        lazzy: bool = False,
    ) -> Union[List[Model], Tuple[List[Model], int]]:
        query = session.query(cls.model)
        if fuzzy_search:
            query = cls._fuzzy_search(query, fuzzy_search, session=session)
        query = cls.apply_domain(query, domain, session)
        if "active" in cls.model.__table__.c:
            active_filter = cls.model.active == active
            active_filter = (
                or_(active_filter, cls.model.active is None) if active else active_filter
            )

            query = query.filter(active_filter)
        if not order_by:
            order_by = cls._get_default_order_by(session=session)
        if need_count:
            count = query.count()
        order_by = cls._normalize_order_by(cls.model, order_by)
        query: Query = query.order_by(text(order_by))
        if lazzy:
            return query
        if limit is not None:
            offset = offset or 0
            query = query.offset(offset).limit(limit)
        records = query.all()
        cls.ensure_role_access(records, session=session, context=context)
        return (records, count) if need_count else records

    @classmethod
    @add_session
    def search(
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
    ) -> SearchResultPaged:
        next_page = False
        records, total_records = cls._search(
            domain,
            order_by,
            limit,
            offset,
            active,
            fuzzy_search,
            session=session,
            context=context,
            need_count=True,
        )
        if limit and len(records) > limit:
            records.pop()
            next_page = True
        return records, next_page, total_records

    @classmethod
    @add_session
    def all_ids(cls, *, session=None) -> Set[int]:
        records = session.query(cls.model.id).all()
        return {record[0] for record in records}

    @classmethod
    def _check_data_key_value(cls, key, value):
        if key in cls.restricted_fields:
            raise ForbiddenError(f"The field '{key}' can not be set manually")
        if key in cls.pseudo_enums and value not in cls.pseudo_enums[key]:
            raise ForbiddenError(f"The field '{key}' don not supports '{value}' value")

    @classmethod
    @add_session
    def _check_to_update_data(cls, data: Dict[str, Any], *, session=None, context=None):
        for key in data:
            if key in cls.restricted_update_fields:
                raise ForbiddenError(f"The field '{key}' can not be updated manually")

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
        if is_super_user(context):
            return
        for key, value in data.items():
            cls._check_data_key_value(key, value)

    @classmethod
    @add_session
    @check_context
    def create(cls, data: Dict[str, Any], *, session=None, context=None):
        data["identifier"] = data.get("identifier", identifier_default_factory())
        try:
            m2m = []
            for key, value in data.copy().items():
                if m2m_rel := is_x2m(cls.model, key):
                    data.pop(key)
                    m2m.append((m2m_rel, key, value))
                    continue
            record = cls.model(**data)
            for rel, key, value in m2m:
                field = getattr(record, key)
                cls._set_m2m(rel.property.entity, field, value, session=session)  # type: ignore
        except TypeError as e:
            raise ForbiddenError(e) from e
        cls._check_data(record, data, session=session, context=context)
        session.add(record)
        return record

    @staticmethod
    def _to_primitive(data) -> PrimitiveType:
        def get_class(field):
            return enum.Enum if issubclass(field.__class__, enum.Enum) else field.__class__

        value_class = get_class(data)

        converters = {
            datetime: lambda x: x.isoformat(),
            date: lambda x: x.isoformat(),
            enum.Enum: lambda x: x.name,
            Identifier: lambda x: str(x),  # pylint: disable=unnecessary-lambda
        }
        return converters.get(value_class, lambda x: x)(data)

    @classmethod
    def record_to_dict(cls, record: Model, fields_string: Set["str"]) -> Dict[str, Any]:
        """Return a dictionary with the fields given (can use dot to indicate subfields)
        filled with the object data

        Args:
            record (Model): Record to convert
            fields_string (Set[): Fields to retrieve

        Returns:
            Dict[str, Any]: Dictionary with the fields filled with the object data
        """

        def tokenize(fields: Set[str]) -> Dict[str, Any]:
            """Split the fields using the dot character, creating a dict"""
            result: Dict[str, Any] = {}
            for field in fields:
                parts = field.split(".")
                current = result
                for part in parts:
                    current = current.setdefault(part, {})
            return result

        controllers_by_model = cls.get_controllers_by_model()

        def obj_to_dict(record, tokens: Dict[str, Any]):
            """Create a dictionary based on the token fields"""
            if record is None:
                return None
            result = {}
            for key, value in tokens.items():
                real_value = getattr(record, key)
                m2m = is_x2m(record, key)
                if value == {} and not isinstance(real_value, Model) and not m2m:
                    real_value = cls._to_primitive(real_value)
                    result[key] = real_value
                else:
                    controller = controllers_by_model.get(real_value.__class__, cls)
                    value.update({f: {} for f in controller.default_read_fields})
                    if m2m or isinstance(real_value, list):
                        result[key] = [obj_to_dict(x, value) for x in real_value]
                    else:
                        result[key] = obj_to_dict(getattr(record, key), value)
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
        if not_allowed_companies := company_ids - allowed_companies_ids:
            raise UnauthorizedError(f"Companies `{not_allowed_companies}` not allowed")

    @staticmethod
    def get_model_name_from_records(records: List[Model]) -> str:
        return records[0].__class__.__name__

    @classmethod
    @ensure_list
    def log_records(cls, records):
        if not records:
            return ""
        model_name = cls.get_model_name_from_records(records)
        return f"{model_name}({', '.join([str(r.id) for r in records])})"

    @classmethod
    @add_session
    @check_context
    @ensure_set
    def get(cls, ids: Set[int], *, session=None, context=None, singleton=True) -> List[Model]:
        records = (
            session.query(cls.model).filter(cls.model.id.in_(ids)).order_by(cls.model.id).all()
        )
        if len(ids) != len(records):
            ids_read = {record.id for record in records}
            diff = ids - ids_read
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
    def _remove_from_m2m(cls, model, field, records: List[Model], *, session=None):
        for record in records:
            if record in field:
                field.remove(record)
            else:
                raise NotFoundError(cls.log_records(record))

    @classmethod
    @add_session
    def _delete_m2m_rel(cls, model, field, records: List[Model], *, session=None):
        for record in records:
            if record in field:
                session.delete(record)
            else:
                raise NotFoundError(cls.log_records(record))

    @classmethod
    @add_session
    def _add_to_m2m(cls, model, field, records: List[Model], *, session=None):
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
    def _replace_all_from_m2m(
        cls, model, field, records: Union[Model, List[Model]], *, session=None
    ):
        field.clear()
        records = records if isinstance(records, list) else [records]
        field.extend(records)

    @classmethod
    @add_session
    def _set_m2m(
        cls, model, field, value: List[Tuple[int, Union[int, None, List[int]]]], *, session=None
    ):
        """Update an m2m field based on the next structure:
        (0, None): NotImplemented
        (1, id): NotImplemented
        (2, id): Remove but NOT delete from the DB
        (3, id): Remove and delete from the DB
        (4, id): Add in relation
        (5, None): Remove all ids from the relation
        (6, ids): Replace all current ids with the provided ids

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
    def _onchange_fields(cls, fields: List[str], record: Model, *, session=None, context=None):
        functions: Set[Callable] = set()
        for field in fields:
            for function in cls.onchange_functions.get(field, set()):
                functions.add(function)
        for function in functions:
            callable_function = function.__get__(object)  # type: ignore
            callable_function(record, session=session, context=context)

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
                if m2m_rel := is_x2m(record, key):
                    field = getattr(record, key)
                    cls._set_m2m(m2m_rel.property.entity, field, value, session=session)
                    continue
                setattr(record, key, value)
            cls._onchange_fields(list(data.keys()), record, session=session, context=context)
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

    @staticmethod
    def to_csv(query: Query, fields: List[str], session, context) -> bytes:
        f = io.StringIO()
        writer = csv.writer(f)
        writer.writerow(fields)
        for record in query:
            writer.writerow([_plain_field(record, field) for field in fields])
        return f.getvalue().encode("utf-8")

    @staticmethod
    def to_xlsx(query: Query, fields: List[str], resume, session, context) -> bytes:
        wb = Workbook()
        ws = wb.active
        ws.title = "Cfdis"
        ws.append(fields)
        for record in query:
            data = [_plain_field(record, field) for field in fields]
            ws.append(data)
        for column_cells in ws.columns:
            length = max(len(str(cell.value)) for cell in column_cells)
            ws.column_dimensions[column_cells[0].column_letter].width = length * 1.1  # Magic Number
        ws2 = wb.create_sheet("Totales")
        ws2.append(resume_fields)
        if resume["filtered"]:
            filtered = [
                "Periodo",
                resume["filtered"]["count"],
                resume["filtered"]["RetencionesIVA"],
                resume["filtered"]["RetencionesIEPS"],
                resume["filtered"]["RetencionesISR"],
                resume["filtered"]["TrasladosIVA"],
                resume["filtered"]["TrasladosIEPS"],
                resume["filtered"]["TrasladosISR"],
                resume["filtered"]["ImpuestosRetenidos"],
                resume["filtered"]["SubTotal"],
                resume["filtered"]["Descuento"],
                resume["filtered"]["Neto"],
                resume["filtered"]["Total"],
            ]

            ws2.append(filtered)

        if resume["excercise"]:
            excercise = [
                "Acumulado",
                resume["excercise"]["count"],
                resume["excercise"]["RetencionesIVA"],
                resume["excercise"]["RetencionesIEPS"],
                resume["excercise"]["RetencionesISR"],
                resume["excercise"]["TrasladosIVA"],
                resume["excercise"]["TrasladosIEPS"],
                resume["excercise"]["TrasladosISR"],
                resume["excercise"]["ImpuestosRetenidos"],
                resume["excercise"]["SubTotal"],
                resume["excercise"]["Descuento"],
                resume["excercise"]["Neto"],
                resume["excercise"]["Total"],
            ]

            ws2.append(excercise)

        for column_cells in ws2.columns:
            length = max(len(str(cell.value)) for cell in column_cells)
            ws2.column_dimensions[column_cells[0].column_letter].width = (
                length * 1.1
            )  # Magic Number

        with NamedTemporaryFile(suffix="xlsx") as f:
            wb.save(f.name)
            with open(f.name, "rb") as f2:
                return f2.read()

    @staticmethod
    def get_xml(records: List[Model]) -> List[Dict[str, str]]:
        ...

    @staticmethod
    def to_xml(query: Query, _fields: List[str], session, context) -> bytes:
        """Return a ZIP with the XML's of the records"""
        controllers_by_model = CommonController.get_controllers_by_model()
        controller = controllers_by_model[query[0].__class__]
        urls = controller.get_xml(query.all())
        f = io.BytesIO()
        with ZipFile(f, "w") as zf:
            for row in urls:
                uuid, url = row["uuid"], row["xml_url"]
                xml = requests.get(url).content  # TODO async
                zf.writestr(f"{uuid}.xml", xml)
        return f.getvalue()

    @classmethod
    def to_pdf(cls, query: Query, _fields: List[str], session, context) -> bytes:
        """Return a ZIP with the XML's of the records"""
        f = io.BytesIO()
        with ZipFile(f, "w") as zf:
            for record in query:
                pdf = cls._to_pdf(record)  # TODO async
                zf.writestr(f"{record.UUID}.pdf", pdf)
        return f.getvalue()

    @staticmethod
    def _to_pdf(record: Model) -> bytes:
        ...

    @classmethod
    @add_session
    def export(
        cls,
        query: Query,
        fields: List[str],
        export_str: str,
        resume_export=None,
        *,
        session,
        context,
    ) -> Dict[str, str]:
        export_format = ExportFormat[export_str]
        EXPORTERS = {
            ExportFormat.CSV: cls.to_csv,
            ExportFormat.XLSX: cls.to_xlsx,
            ExportFormat.XML: cls.to_xml,
            ExportFormat.PDF: cls.to_pdf,
        }
        exporter = EXPORTERS.get(export_format)
        extension = {
            "CSV": "csv",
            "XLSX": "xlsx",
            "XML": "zip",
            "PDF": "zip",
        }[export_str]
        if not exporter:
            raise NotFoundError(f"Export format {export_format} not implemented")
        _logger.info("Exporting records")
        if not query.count():
            raise NotFoundError("No records found")
        data_bytes = None
        if export_str in ["XLSX", "xlsx"]:
            data_bytes = exporter(query, fields, resume_export, session, context)
        else:
            data_bytes = exporter(query, fields, session, context)
        model_name = cls.model.__name__
        now = utc_now()
        date_str = now.strftime("%Y_%m_%d_%H_%M_%S")
        filename = f"{model_name}_{date_str}.{extension}"

        s3_client = boto3.client("s3")
        _logger.info("Uploading to S3")
        s3_client.upload_fileobj(  # TODO deal with collisions
            io.BytesIO(data_bytes),
            envars.S3_EXPORT,
            filename,
        )
        _logger.info("Uploaded to S3")
        s3_url = s3_client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": envars.S3_EXPORT,
                "Key": filename,
            },
            ExpiresIn=EXPORT_EXPIRATION,
        )
        return {
            "url": s3_url,
        }

    @staticmethod
    @add_session
    def resume(domain: Domain, fuzzy_search: str = None, *, session=None, context):
        raise MethodNotAllowedError("Resume not implemented")


def get_m2m_repr(m2m_rel, attributes: Set[str]) -> List[Dict[str, Any]]:
    return [
        {
            attrib: getattr(
                rel,
                attrib,
            )
            for attrib in attributes
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
