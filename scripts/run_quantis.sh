#!/bin/bash
#
# Run script for Quantis project.
# Starts the FastAPI backend and the web frontend for local development.
#
# Usage: ./scripts/run_quantis.sh
# Must be run from the repository root (or it will locate it automatically).

set -uo pipefail

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Resolve the repository root regardless of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || { echo -e "${RED}Failed to reach repository root${NC}"; exit 1; }

echo -e "${BLUE}Starting Quantis application...${NC}"

BACKEND_DIR="${REPO_ROOT}/code/backend"
FRONTEND_DIR="${REPO_ROOT}/web-frontend"

if [ ! -d "${BACKEND_DIR}" ]; then
  echo -e "${RED}Backend directory not found at ${BACKEND_DIR}${NC}"
  exit 1
fi
if [ ! -d "${FRONTEND_DIR}" ]; then
  echo -e "${RED}Frontend directory not found at ${FRONTEND_DIR}${NC}"
  exit 1
fi

# Create Python virtual environment if it doesn't exist
if [ ! -d "${REPO_ROOT}/venv" ]; then
  echo -e "${BLUE}Creating Python virtual environment...${NC}"
  python3 -m venv "${REPO_ROOT}/venv" || { echo -e "${RED}Failed to create virtual environment${NC}"; exit 1; }
fi

# shellcheck source=/dev/null
source "${REPO_ROOT}/venv/bin/activate" || { echo -e "${RED}Failed to activate virtual environment${NC}"; exit 1; }

echo -e "${BLUE}Installing backend dependencies...${NC}"
pip install -q -r "${BACKEND_DIR}/requirements.txt" || {
  echo -e "${RED}Failed to install backend dependencies${NC}"
  exit 1
}

# The API needs a signing secret; generate an ephemeral one for local runs
# if the operator hasn't already exported one.
export SECRET_KEY="${SECRET_KEY:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')}"
export JWT_SECRET="${JWT_SECRET:-${SECRET_KEY}}"
export PYTHONPATH="${REPO_ROOT}"

# Start API server (module path matches code/backend/core/app.py's FastAPI instance)
echo -e "${BLUE}Starting API server...${NC}"
python3 -m uvicorn code.backend.core.app:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Start frontend
echo -e "${BLUE}Starting frontend...${NC}"
(
  cd "${FRONTEND_DIR}" || exit 1
  npm install --no-audit --no-fund > /dev/null
  npm run dev
) &
FRONTEND_PID=$!

# Handle graceful shutdown before waiting, so Ctrl+C during the readiness
# wait below is still handled correctly.
function cleanup {
  echo -e "\n${BLUE}Stopping services...${NC}"
  kill "${FRONTEND_PID}" 2>/dev/null
  kill "${API_PID}" 2>/dev/null
  wait "${FRONTEND_PID}" 2>/dev/null
  wait "${API_PID}" 2>/dev/null
  echo -e "${GREEN}All services stopped${NC}"
  exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for the backend to actually be ready instead of guessing with a fixed
# sleep, so we don't report success while the API is still starting up.
echo -e "${BLUE}Waiting for the API to become ready...${NC}"
READY=0
for _ in $(seq 1 30); do
  if curl -fsS "http://localhost:8000/health" > /dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if [ "${READY}" -ne 1 ]; then
  echo -e "${RED}API did not become ready in time. Check the logs above.${NC}"
fi

echo -e "${GREEN}Quantis application is running!${NC}"
echo -e "${GREEN}API running with PID: ${API_PID} — http://localhost:8000${NC}"
echo -e "${GREEN}Frontend running with PID: ${FRONTEND_PID} — http://localhost:3000${NC}"
echo -e "${BLUE}Press Ctrl+C to stop all services${NC}"

# Keep script running until interrupted; if either child exits on its own,
# clean up the other instead of leaving it orphaned.
wait -n "${API_PID}" "${FRONTEND_PID}"
cleanup
