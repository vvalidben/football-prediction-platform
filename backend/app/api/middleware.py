"""Cross-cutting request handling: a request id, one access log line, and headers.

Every response carries `X-Request-ID` — the caller's if it sent one, otherwise a
fresh one — and every request is logged once with its method, path, status and
duration. When something fails, the id in the client's response is the id in the
logs, which is what makes a production report actionable.

Nothing here logs a query string, a body or a header: those carry provider keys
and, in other deployments, personal data.
"""

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("app.access")

REQUEST_ID_HEADER = "X-Request-ID"
#: A caller-supplied id is echoed back only if it looks like one: it ends up in logs.
MAX_REQUEST_ID_LENGTH = 64

#: Applied to every API response. The API serves JSON to a browser app, so the
#: useful ones are those that stop a response being treated as something else.
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Cache-Control": "no-store",
}

Handler = Callable[[Request], Awaitable[Response]]


def _incoming_id(request: Request) -> str | None:
    value = request.headers.get(REQUEST_ID_HEADER)
    if value is None:
        return None
    cleaned = value.strip()[:MAX_REQUEST_ID_LENGTH]
    # Only printable ASCII without control characters, so a log line cannot be forged.
    return cleaned if cleaned and cleaned.isascii() and cleaned.isprintable() else None


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, log the request once, and set the response headers."""

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        request_id = _incoming_id(request) or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # The exception handler turns this into a 500; log the timing here so a
            # failed request is not missing from the access log.
            duration_ms = (time.perf_counter() - started) * 1000
            logger.warning(
                "%s %s failed after %.1f ms [request_id=%s]",
                request.method,
                request.url.path,
                duration_ms,
                request_id,
            )
            raise
        duration_ms = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        logger.log(
            logging.WARNING if response.status_code >= 500 else logging.INFO,
            "%s %s -> %s in %.1f ms [request_id=%s]",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )
        return response


def request_id_of(request: Request) -> str:
    """The current request's id, for error responses."""
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else "unknown"
