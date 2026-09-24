"""Football prediction platform backend."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("football-prediction-backend")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled tree
    __version__ = "0.0.0"
