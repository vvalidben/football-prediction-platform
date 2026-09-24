#!/bin/sh
# Apply database migrations, then hand off to the container command (uvicorn).
set -eu

if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
    echo "Applying database migrations..."
    alembic upgrade head
fi

exec "$@"
