"""End-to-end smoke test for the running Docker Compose stack.

Uses only the standard library, so it runs with any Python 3.12+:

    python tests/smoke/check_stack.py
    python tests/smoke/check_stack.py --backend-url http://127.0.0.1:8000 \
        --frontend-url http://127.0.0.1:8080

Exits non-zero if any check fails.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class HttpResult:
    status: int
    body: str


def http_get(url: str, timeout: float = 10.0) -> HttpResult:
    request = urllib.request.Request(url, headers={"Accept": "*/*"})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return HttpResult(response.status, response.read().decode())
    except urllib.error.HTTPError as error:
        return HttpResult(error.code, error.read().decode())


def expect_healthy(url: str) -> str:
    result = http_get(url)
    if result.status != 200:
        raise AssertionError(f"expected HTTP 200, got {result.status}: {result.body[:200]}")
    report = json.loads(result.body)
    failing = {name: c for name, c in report["checks"].items() if c["status"] != "ok"}
    if report["status"] != "ok" or failing:
        raise AssertionError(f"degraded: {failing}")
    latencies = ", ".join(f"{n}={c['latency_ms']}ms" for n, c in report["checks"].items())
    return f"{report['service']} v{report['version']} ({latencies})"


def expect_page(url: str) -> str:
    result = http_get(url)
    if result.status != 200 or '<div id="root">' not in result.body:
        raise AssertionError(f"expected the SPA shell, got HTTP {result.status}")
    return "SPA shell served"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--frontend-url", default="http://127.0.0.1:8080")
    args = parser.parse_args()

    checks: list[tuple[str, Callable[[], str]]] = [
        ("backend /health", lambda: expect_healthy(f"{args.backend_url}/health")),
        ("frontend index", lambda: expect_page(f"{args.frontend_url}/")),
        ("frontend SPA fallback", lambda: expect_page(f"{args.frontend_url}/any/client/route")),
        (
            "frontend -> backend proxy",
            lambda: expect_healthy(f"{args.frontend_url}/api/health"),
        ),
    ]

    failures = 0
    for name, check in checks:
        try:
            print(f"PASS  {name}: {check()}")
        except (AssertionError, OSError, ValueError, KeyError) as error:
            failures += 1
            print(f"FAIL  {name}: {error}")

    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
