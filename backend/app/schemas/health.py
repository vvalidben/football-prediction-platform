from typing import Literal

from pydantic import BaseModel

CheckStatus = Literal["ok", "error"]
OverallStatus = Literal["ok", "degraded"]


class DependencyCheck(BaseModel):
    status: CheckStatus
    latency_ms: float | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: OverallStatus
    service: str
    version: str
    environment: str
    checks: dict[str, DependencyCheck]
