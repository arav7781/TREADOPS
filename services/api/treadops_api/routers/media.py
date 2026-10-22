"""Short-lived signed media links. The token is the capability; storage credentials never leave the server."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response

from ..errors import ApiError
from ..security import verify_media
from ..storage import StorageUnavailable, content_type_for, get_storage

router = APIRouter(prefix="/v1", tags=["media"])


@router.get("/media/{token}")
def media(token: str):
    key = verify_media(token)
    try:
        data = get_storage().get(key)
    except FileNotFoundError as exc:
        raise ApiError(404, "not_found", "Media object not found.") from exc
    except StorageUnavailable as exc:
        raise ApiError(503, "storage_unavailable", "Object storage is unavailable.", retry_after=10) from exc
    return Response(data, media_type=content_type_for(key), headers={"Cache-Control": "private, max-age=300"})
