"""Validate API-Football as the primary data provider (Sprint 2).

    python scripts/validate_api_football.py [--budget 25]

The work lives with the provider it validates
(`backend/app/providers/api_football/validation.py`), so it reuses the provider
abstraction, its HTTP client and its error types rather than duplicating them.
This script only makes that runnable from the repository root.

The API key is read from the environment (`API_FOOTBALL_API_KEY`, the variable this
project already uses) and is never printed, logged or written to the artefacts. If
it is missing, the run stops and says exactly how to configure it.

Outputs:
    artifacts/api_football_validation.json    machine-readable findings
    docs/api-football-validation.md           the report
"""

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.providers.api_football.validation import main  # noqa: E402  - after sys.path

if __name__ == "__main__":
    sys.exit(main())
