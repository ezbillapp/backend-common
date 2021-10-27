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
    attr_list = {
        "fuzzy_search": None,
        "domain": {},
        "order_by": None,
        "limit": PAGE_SIZE,
        "offset": None,
        "active": True,
    }
    return {attr: json_body.get(attr, default) for attr, default in attr_list.items()}


def search(bp, controller):
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
        raise Exception("No token provided")

    pos, next_page = controller.search(**search_attrs, context=context)
    dict_repr = controller.detail(pos, fields, context=context)
    return {
        "data": dict_repr,
        "next_page": next_page,
    }


def create(bp, controller):
    json_body = bp.current_request.json_body or {}
    token = bp.current_request.headers["access_token"]

    user = UserController.get_by_token(token)
    context = {"user": user}
    po = controller.create(
        json_body, context=context
    )  # TODO use `data` section and allow list of values
    dict_repr = controller.detail(po, context=context)  # TODO add `fields`
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
    return controller.detail(pos, context=context)  # TODO: add `fields`


def delete(bp, controller):
    json_body = bp.current_request.json_body or {}
    token = bp.current_request.headers["access_token"]

    user = UserController.get_by_token(token)
    context = {"user": user}
    ids = set(json_body["ids"])
    pos = controller.get(ids, context=context)
    ids = controller.delete(pos, context=context)
    return {"deleted": list(ids)}
