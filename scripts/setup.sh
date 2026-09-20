#!/bin/bash

set -euo pipefail

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ".env created from .env.example"
fi

if grep -q '^ENV=' .env 2>/dev/null; then
    sed -i '' '/^ENV=/d' .env
fi

echo "Reading .env..."

set -a
source .env
set +a

is_true() {
    case "${1:-}" in
        true|TRUE|True|yes|YES|Yes|1|on|ON|On)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}
# ------------------------------------------------------------
# Python environment
# ------------------------------------------------------------

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate


# ------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------

echo "Installing Python dependencies..."

python -m pip install -r requirements.txt


# ------------------------------------------------------------
# YOLO model
# ------------------------------------------------------------

echo "Downloading model..."

# Just the active model - `make start` should stay fast; use
# `make model_download` (no args) to fetch every registered model.
python scripts/download_model.py --active


# ------------------------------------------------------------
# Docker
# ------------------------------------------------------------

echo "Checking Docker..."

if ! docker info >/dev/null 2>&1; then
    echo ""
    echo "ERROR: Docker is not running."
    echo ""
    echo "Please start Docker Desktop and run:"
    echo ""
    echo "    make start"
    echo ""
    exit 1
fi

echo "Docker is running."


# ------------------------------------------------------------
# MongoDB
# ------------------------------------------------------------

echo "Checking MongoDB..."

if docker ps --format '{{.Names}}' | grep -q '^test-mongo$'; then

    echo "MongoDB container is already running."

else

    if docker ps -a --format '{{.Names}}' | grep -q '^test-mongo$'; then

        echo "Starting existing MongoDB container..."
        docker start test-mongo >/dev/null

    else

        if lsof -i :"$MONGO_PORT" >/dev/null 2>&1; then
            echo ""
            echo "ERROR: Port $MONGO_PORT is already in use."
            echo "Please stop the service using that port or update MONGO_PORT in .env."
            echo ""
            exit 1
        fi

        echo "Creating MongoDB container..."

        docker run -d \
            --name=test-mongo \
            -p "$MONGO_PORT":27017 \
            mongo:latest >/dev/null

    fi

fi


# ------------------------------------------------------------
# Wait for MongoDB
# ------------------------------------------------------------

echo "Waiting for MongoDB..."

until docker exec test-mongo mongosh \
    --eval "db.adminCommand('ping')" \
    >/dev/null 2>&1
do
    sleep 2
done

echo "MongoDB ready."


# ------------------------------------------------------------
# Optional MongoDB reset
# ------------------------------------------------------------

if [ "${STARTUP_MODE:-}" = "start" ] && is_true "${CLEAR_MONGO_DB:-false}"; then
    echo "Clearing MongoDB database: $MONGO_DB"
    docker exec test-mongo mongosh \
        --quiet \
        --eval "db.getSiblingDB('${MONGO_DB}').dropDatabase()" \
        >/dev/null
    echo "MongoDB database cleared."
fi


# ------------------------------------------------------------
# Optional model registry reset
# ------------------------------------------------------------

if [ "${STARTUP_MODE:-}" = "start" ] && is_true "${CLEAR_MODEL_REGISTRY:-false}"; then
    echo "Clearing model registry dataset-version history and cached performance..."
    PYTHONPATH=. python -m model_management.model_registry.registry reset --yes
fi


# ------------------------------------------------------------
# Setup complete
# ------------------------------------------------------------

echo ""
echo "========================================"
echo "Setup complete."
echo "========================================"
echo ""

echo "Run application using:"
echo "make start"

echo ""