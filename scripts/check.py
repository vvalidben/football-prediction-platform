"""Run every linter, type checker and test suite for the project.

    python scripts/check.py                 # static checks + unit tests
    python scripts/check.py --integration   # also backend integration tests
                                            # (needs PostgreSQL + Redis running)
    python scripts/check.py --audit         # also dependency vulnerability audits
                                            # (needs network access)

Expects the backend virtualenv at backend/.venv and frontend dependencies
installed (see README.md, "Local development"). Exits non-zero on the first
failing step.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
VENV_BIN = BACKEND / ".venv" / ("Scripts" if os.name == "nt" else "bin")


def venv_tool(name: str) -> str:
    return str(VENV_BIN / name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration", action="store_true", help="run integration tests too")
    parser.add_argument(
        "--audit",
        action="store_true",
        help="also audit dependencies for known vulnerabilities (needs network access)",
    )
    args = parser.parse_args()

    npm = shutil.which("npm")
    if npm is None:
        print("npm not found on PATH", file=sys.stderr)
        return 1
    if not VENV_BIN.exists():
        print(f"backend virtualenv not found at {VENV_BIN.parent}", file=sys.stderr)
        return 1

    pytest_env = {**os.environ, "RUN_INTEGRATION_TESTS": "1" if args.integration else "0"}
    steps: list[tuple[str, list[str], Path, dict[str, str] | None]] = [
        ("backend: ruff format", [venv_tool("ruff"), "format", "--check", "."], BACKEND, None),
        ("backend: ruff lint", [venv_tool("ruff"), "check", "."], BACKEND, None),
        ("backend: mypy", [venv_tool("mypy")], BACKEND, None),
        ("backend: pytest", [venv_tool("pytest")], BACKEND, pytest_env),
        (
            "tooling: ruff",
            [venv_tool("ruff"), "check", "--config", "backend/pyproject.toml", "scripts", "tests"],
            ROOT,
            None,
        ),
        ("frontend: prettier", [npm, "run", "format:check"], FRONTEND, None),
        ("frontend: eslint", [npm, "run", "lint"], FRONTEND, None),
        ("frontend: typecheck", [npm, "run", "typecheck"], FRONTEND, None),
        ("frontend: vitest", [npm, "test"], FRONTEND, None),
        ("frontend: build", [npm, "run", "build"], FRONTEND, None),
    ]

    if args.audit:
        # Kept opt-in: both reach the network, so an offline run of the suite
        # should not fail for a reason that has nothing to do with the code.
        steps += [
            (
                "backend: pip-audit",
                [venv_tool("python"), "-m", "pip_audit", "--progress-spinner", "off"],
                BACKEND,
                None,
            ),
            ("frontend: npm audit", [npm, "audit", "--audit-level", "high"], FRONTEND, None),
        ]

    for name, command, cwd, env in steps:
        print(f"\n=== {name}", flush=True)
        if subprocess.run(command, cwd=cwd, env=env, check=False).returncode != 0:  # noqa: S603
            print(f"\nFAILED: {name}", file=sys.stderr)
            return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
