from chalice import CORSConfig

from ..config import PAGE_SIZE
from ..controllers.user import UserController

cors_config = CORSConfig(
    allow_origin="*",
    allow_headers=["access_token"],
    max_age=None,
    expose_headers=None,
    allow_credentials=None,
)


def get_search_attrs(json_body):
    fuzzy_search = json_body.get("fuzzy_search", None)
    domain = json_body.get("domain", {})
    order_by = json_body.get("order_by")
    limit = json_body.get("limit", PAGE_SIZE)
    offset = json_body.get("offset")
    active = json_body.get("active", True)
    return {
        "fuzzy_search": fuzzy_search,
        "domain": domain,
        "order_by": order_by,
        "limit": limit,
        "offset": offset,
        "active": active,
    }


def search(bp, controller):
    json_body = bp.current_request.json_body or {}
    headers = bp.current_request.headers
    token = headers.get("access_token")
    temporal_token = headers.get("temporal_token")

    search_attrs = get_search_attrs(json_body)

    if token:
        user = UserController.get_by_token(token)
        context = {"user": user}
    elif temporal_token:
        context = {"guest_partner": True}
    else:
        raise Exception("No token provided")

    pos, next_page = controller.search(**search_attrs, context=context)
    dict_repr = controller.detail(pos, context=context)
    return {
        "data": dict_repr,
        "next_page": next_page,
    }


def create(bp, controller):
    json_body = bp.current_request.json_body or {}
    token = bp.current_request.headers["access_token"]

    user = UserController.get_by_token(token)
    context = {"user": user}
    po = controller.create(json_body, context=context)
    dict_repr = controller.detail(po, context=context)
    return dict_repr[0]


def update(bp, controller):
    json_body = bp.current_request.json_body or {}
    token = bp.current_request.headers["access_token"]

    ids = set(json_body["ids"])
    values = json_body["values"]
    user = UserController.get_by_token(token)
    context = {"user": user}
    pos = controller.get(ids, context=context)
    controller.update(pos, values, context=context)
    return controller.detail(pos, context=context)


def delete(bp, controller):
    json_body = bp.current_request.json_body or {}
    token = bp.current_request.headers["access_token"]

    user = UserController.get_by_token(token)
    context = {"user": user}
    ids = set(json_body["ids"])
    pos = controller.get(ids, context=context)
    ids = controller.delete(pos, context=context)
    return {"deleted": list(ids)}
