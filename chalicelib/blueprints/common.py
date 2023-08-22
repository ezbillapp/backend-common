from chalice import CORSConfig, UnauthorizedError

from chalicelib.config import PAGE_SIZE
from chalicelib.controllers.common import CommonController
from chalicelib.controllers.user import UserController
from chalicelib.new.config.infra.log import logger

cors_config = CORSConfig(
    allow_origin="*",
    allow_headers=["access_token"],
    max_age=None,
    expose_headers=None,
    allow_credentials=None,
)


def get_search_attrs(json_body):
    attr_list = {
        "fuzzy_search": None,
        "domain": {},
        "order_by": None,
        "limit": PAGE_SIZE,
        "offset": None,
        "active": True,
    }
    return {attr: json_body.get(attr, default) for attr, default in attr_list.items()}


def export(bp, controller: CommonController):
    json_body = bp.current_request.json_body or {}
    headers = bp.current_request.headers
    token = headers.get("access_token")
    temporal_token = headers.get("temporal_token")

    search_attrs = get_search_attrs(json_body)
    search_attrs["limit"] = None
    search_attrs["offset"] = None
    fields = json_body.get("fields", [])
    fields = tuple(fields)
    export_format = json_body.get("format", "csv")
    resume_export = None
    if export_format in ["xlsx", "XLSX"]:
        resume_export = resume(bp, controller)

    if token:
        user = UserController.get_by_token(token)
        context = {"user": user}
    elif temporal_token:
        context = {"guest_partner": True}
    else:
        raise UnauthorizedError("No token provided")

    query = controller._search(  # pylint: disable=protected-access
        **search_attrs, context=context, lazzy=True
    )
    return controller.export(query, fields, export_format, resume_export, context=context)


def massive_export(bp, controller: CommonController):
    json_body = bp.current_request.json_body or {}
    headers = bp.current_request.headers
    token = headers.get("access_token")
    temporal_token = headers.get("temporal_token")

    search_attrs = get_search_attrs(json_body)
    search_attrs["limit"] = None
    search_attrs["offset"] = None
    fields = json_body.get("fields", [])
    export_format = json_body.get("format", "csv")

    if token:
        user = UserController.get_by_token(token)
        context = {"user": user}
    elif temporal_token:
        context = {"guest_partner": True}
    else:
        raise UnauthorizedError("No token provided")
    return json_body


def search(bp, controller: CommonController):
    json_body = bp.current_request.json_body or {}
    headers = bp.current_request.headers
    token = headers.get("access_token")
    temporal_token = headers.get("temporal_token")

    search_attrs = get_search_attrs(json_body)
    fields = json_body.get("fields", [])

    if token:
        user = UserController.get_by_token(token)
        context = {"user": user}
    elif temporal_token:
        context = {"guest_partner": True}
    else:
        raise UnauthorizedError("No token provided")

    pos, next_page, total_records = controller.search(**search_attrs, context=context)
    dict_repr = controller.detail(pos, fields, context=context)
    return {
        "data": dict_repr,
        "next_page": next_page,
        "total_records": total_records,
    }


def create(bp, controller: CommonController):
    json_body = bp.current_request.json_body or {}
    token = bp.current_request.headers["access_token"]

    user = UserController.get_by_token(token)
    context = {"user": user}
    po = controller.create(
        json_body, context=context
    )  # TODO use `data` section and allow list of values
    dict_repr = controller.detail(po, context=context)
    return dict_repr[0]


def update(bp, controller: CommonController):
    json_body = bp.current_request.json_body or {}
    token = bp.current_request.headers["access_token"]

    ids = set(json_body["ids"])
    values = json_body["values"]
    user = UserController.get_by_token(token)
    context = {"user": user}
    pos = controller.get(ids, context=context)
    controller.update(pos, values, context=context)
    return controller.detail(pos, context=context)


def delete(bp, controller: CommonController):
    json_body = bp.current_request.json_body or {}
    token = bp.current_request.headers["access_token"]

    user = UserController.get_by_token(token)
    context = {"user": user}
    ids = set(json_body["ids"])
    pos = controller.get(ids, context=context)
    ids = controller.delete(pos, context=context)
    return {"deleted": list(ids)}


def resume(bp, controller: CommonController):
    json_body = bp.current_request.json_body or {}
    token = bp.current_request.headers["access_token"]

    domain = json_body.get("domain", [])
    fuzzy_search = json_body.get("fuzzy_search", [])
    user = UserController.get_by_token(token)
    context = {"user": user}
    return controller.resume(domain, fuzzy_search, context=context)


def get_count_cfdis(bp, controller: CommonController):
    json_body = bp.current_request.json_body or {}
    token = bp.current_request.headers["access_token"]

    domain = json_body.get("domain", [])
    fuzzy_search = json_body.get("fuzzy_search", [])
    user = UserController.get_by_token(token)
    context = {"user": user}
    return controller.count_cfdis_by_type(domain, fuzzy_search, context=context)
