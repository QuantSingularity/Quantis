#!/bin/bash

# Quantis Project Test Script
# This script runs all unit and integration tests for the project.

# Exit immediately if a command exits with a non-zero status, and treat unset variables as an error.
set -uo pipefail

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

# Track overall success across all three suites so we can report a single
# meaningful exit code instead of stopping at the first failure.
OVERALL_STATUS=0

echo -e "${BLUE}Starting Quantis test suite...${NC}"

# --- 1. Python Backend Tests (API) ---
echo "----------------------------------------"
echo -e "${BLUE}Running Python Backend Tests (pytest)...${NC}"

PYTHON_VENV="${PROJECT_ROOT}/venv"
if [ ! -d "${PYTHON_VENV}" ]; then
  echo -e "${RED}Error: Python virtual environment not found at ${PYTHON_VENV}.${NC}"
  echo -e "${RED}Please run the setup script first: ${PROJECT_ROOT}/scripts/setup_quantis_env.sh${NC}"
  OVERALL_STATUS=1
elif [ ! -d "${API_DIR}/tests" ]; then
  echo -e "${RED}Warning: '${API_DIR}/tests' not found. Skipping Python tests.${NC}"
else
  if (
    set -e
    # shellcheck source=/dev/null
    source "${PYTHON_VENV}/bin/activate"

    # Ensure pytest and the coverage plugin are installed independently —
    # a system could have pytest but not pytest-cov, which would otherwise
    # crash the --cov invocation below.
    if ! command -v pytest &> /dev/null; then
      echo "Installing pytest..."
      pip install -q pytest
    fi
    if ! python3 -c "import pytest_cov" &> /dev/null; then
      echo "Installing pytest-cov..."
      pip install -q pytest-cov
    fi

    # Run from the project root so the "code" package resolves correctly.
    cd "${PROJECT_ROOT}"
    export PYTHONPATH="${PROJECT_ROOT}"

    echo "Executing pytest with coverage..."
    pytest --cov=code/backend --cov-report=term-missing "${API_DIR}/tests"
  ); then
    echo -e "${GREEN}Python Backend Tests completed.${NC}"
  else
    echo -e "${RED}Python Backend Tests failed.${NC}"
    OVERALL_STATUS=1
  fi
fi

# --- 2. Web Frontend Tests ---
echo "----------------------------------------"
echo -e "${BLUE}Running Web Frontend Tests...${NC}"

if [ ! -d "${WEB_FRONTEND_DIR}" ]; then
    echo -e "${RED}Warning: Web Frontend directory ${WEB_FRONTEND_DIR} not found. Skipping tests.${NC}"
elif [ ! -f "${WEB_FRONTEND_DIR}/package.json" ]; then
    echo -e "${RED}Warning: package.json not found in ${WEB_FRONTEND_DIR}. Skipping tests.${NC}"
elif ! grep -q '"test":' "${WEB_FRONTEND_DIR}/package.json"; then
    echo -e "${RED}Warning: 'test' script not found in ${WEB_FRONTEND_DIR}/package.json. Skipping tests.${NC}"
else
    echo "Executing Web Frontend tests (npm test)..."
    if (cd "${WEB_FRONTEND_DIR}" && npm test); then
        echo -e "${GREEN}Web Frontend Tests completed.${NC}"
    else
        echo -e "${RED}Web Frontend Tests failed.${NC}"
        OVERALL_STATUS=1
    fi
fi

# --- 3. Mobile Frontend Tests ---
echo "----------------------------------------"
echo -e "${BLUE}Running Mobile Frontend Tests...${NC}"

if [ ! -d "${MOBILE_FRONTEND_DIR}" ]; then
    echo -e "${RED}Warning: Mobile Frontend directory ${MOBILE_FRONTEND_DIR} not found. Skipping tests.${NC}"
elif [ ! -f "${MOBILE_FRONTEND_DIR}/package.json" ]; then
    echo -e "${RED}Warning: package.json not found in ${MOBILE_FRONTEND_DIR}. Skipping tests.${NC}"
elif ! grep -q '"test":' "${MOBILE_FRONTEND_DIR}/package.json"; then
    echo -e "${RED}Warning: 'test' script not found in ${MOBILE_FRONTEND_DIR}/package.json. Skipping tests.${NC}"
else
    echo "Executing Mobile Frontend tests (npm test)..."
    if (cd "${MOBILE_FRONTEND_DIR}" && npm test); then
        echo -e "${GREEN}Mobile Frontend Tests completed.${NC}"
    else
        echo -e "${RED}Mobile Frontend Tests failed.${NC}"
        OVERALL_STATUS=1
    fi
fi

echo "----------------------------------------"
if [ "${OVERALL_STATUS}" -eq 0 ]; then
  echo -e "${GREEN}Quantis test suite finished — all suites passed!${NC}"
else
  echo -e "${RED}Quantis test suite finished with failures. See above for details.${NC}"
fi
exit "${OVERALL_STATUS}"
