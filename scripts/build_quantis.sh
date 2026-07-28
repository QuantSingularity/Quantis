#!/bin/bash

# Quantis Project Build Script
# This script builds the production assets for the API and web frontend,
# and installs mobile frontend dependencies (Expo apps are built via EAS,
# not a local build step — see the note below).

# Exit immediately if a command exits with a non-zero status, and treat unset variables as an error.
set -euo pipefail

# --- Configuration ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_DIR="${PROJECT_ROOT}/code/backend"
WEB_FRONTEND_DIR="${PROJECT_ROOT}/web-frontend"
MOBILE_FRONTEND_DIR="${PROJECT_ROOT}/mobile-frontend"

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Quantis production build process...${NC}"

# --- 1. API Build (Python) ---
echo "----------------------------------------"
echo -e "${BLUE}Building API dependencies...${NC}"
# For Python, "building" means ensuring the virtual environment exists and
# dependencies are installed and up to date with requirements.txt.

PYTHON_VENV="${PROJECT_ROOT}/venv"
if [ ! -d "${API_DIR}" ]; then
  echo -e "${RED}Error: API directory not found at ${API_DIR}.${NC}"
  exit 1
fi

if [ ! -d "${PYTHON_VENV}" ]; then
  echo -e "${RED}Error: Python virtual environment not found at ${PYTHON_VENV}.${NC}"
  echo -e "${RED}Please run the setup script first: ${PROJECT_ROOT}/scripts/setup_quantis_env.sh${NC}"
  exit 1
fi

# shellcheck source=/dev/null
source "${PYTHON_VENV}/bin/activate"
echo "Ensuring API dependencies are up to date..."
pip install -q -r "${API_DIR}/requirements.txt"
echo "Verifying the API package imports cleanly..."
(cd "${PROJECT_ROOT}" && PYTHONPATH="${PROJECT_ROOT}" python3 -c "from code.backend.core.app import app" )
deactivate
echo -e "${GREEN}API dependencies verified.${NC}"

# --- 2. Web Frontend Build ---
echo "----------------------------------------"
echo -e "${BLUE}Building Web Frontend...${NC}"

if [ ! -d "${WEB_FRONTEND_DIR}" ]; then
    echo -e "${RED}Warning: Web Frontend directory ${WEB_FRONTEND_DIR} not found. Skipping build.${NC}"
else
    if [ ! -f "${WEB_FRONTEND_DIR}/package.json" ]; then
        echo -e "${RED}Warning: package.json not found in ${WEB_FRONTEND_DIR}. Skipping build.${NC}"
    else
        echo "Installing/Verifying Web Frontend dependencies..."
        (cd "${WEB_FRONTEND_DIR}" && npm install --no-audit --no-fund)

        echo "Running Web Frontend production build..."
        (cd "${WEB_FRONTEND_DIR}" && npm run build)
        echo -e "${GREEN}Web Frontend build completed. Output in ${WEB_FRONTEND_DIR}/build.${NC}"
    fi
fi

# --- 3. Mobile Frontend ---
echo "----------------------------------------"
echo -e "${BLUE}Preparing Mobile Frontend...${NC}"

if [ ! -d "${MOBILE_FRONTEND_DIR}" ]; then
    echo -e "${RED}Warning: Mobile Frontend directory ${MOBILE_FRONTEND_DIR} not found. Skipping.${NC}"
else
    if [ ! -f "${MOBILE_FRONTEND_DIR}/package.json" ]; then
        echo -e "${RED}Warning: package.json not found in ${MOBILE_FRONTEND_DIR}. Skipping.${NC}"
    else
        echo "Installing Mobile Frontend dependencies (npm)..."
        (cd "${MOBILE_FRONTEND_DIR}" && npm install --no-audit --no-fund)

        echo "Type-checking the Mobile Frontend..."
        (cd "${MOBILE_FRONTEND_DIR}" && npm run typecheck)

        echo -e "${GREEN}Mobile Frontend dependencies installed and type-checked.${NC}"
        echo "Expo apps do not have a local 'npm run build' step. To produce"
        echo "installable native binaries, use EAS Build from within ${MOBILE_FRONTEND_DIR}:"
        echo "  npx eas-cli build --platform all"
        echo "(requires an Expo account: https://docs.expo.dev/build/introduction/)"
        echo "For a static web export instead, run: npx expo export --platform web"
    fi
fi

echo "----------------------------------------"
echo -e "${GREEN}Quantis production build process finished!${NC}"
echo "The API and Web Frontend are ready for deployment."
echo "The Mobile Frontend is ready for an EAS Build / web export."
