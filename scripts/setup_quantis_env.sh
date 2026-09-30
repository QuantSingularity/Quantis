#!/bin/bash
#
# Quantis Project Setup Script (Comprehensive)
#
# Installs dependencies for the backend API, web frontend, and mobile
# frontend. Safe to re-run - each section is independent and best-effort
# (a missing/optional component is skipped with a warning rather than
# aborting the whole setup).

set -uo pipefail

# Prerequisites (ensure these are installed):
# - Python 3.9+ and pip
# - Node.js 18+ and npm
# - Docker and Docker Compose (optional, for containerized deployment)

echo "Starting Quantis project setup..."

# Resolve the repository root dynamically instead of assuming a fixed
# install location, so this script works regardless of where the repo was
# cloned/extracted.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}" || exit 1
echo "Using project root: $(pwd)"

# Prefer python3.11 if present (matches CI), otherwise fall back to
# whatever python3 is available so this doesn't fail on other valid
# Python 3 installations.
if command -v python3.11 &> /dev/null; then
  PYTHON_BIN="python3.11"
elif command -v python3 &> /dev/null; then
  PYTHON_BIN="python3"
else
  echo "Error: no python3 interpreter found on PATH. Cannot set up the backend."
  PYTHON_BIN=""
fi

# --- Backend API Setup (FastAPI/Python) ---
echo ""
echo "Setting up Quantis Backend API..."
API_DIR="${PROJECT_DIR}/code/backend"

if [ ! -d "${API_DIR}" ]; then
    echo "Error: Backend API directory ${API_DIR} not found. Skipping API setup."
elif [ -z "${PYTHON_BIN}" ]; then
    echo "Skipping API setup: no Python interpreter available."
else
    (
      cd "${API_DIR}" || exit 1
      echo "Changed directory to $(pwd) for API setup."

      if [ ! -f "requirements.txt" ]; then
          echo "Error: requirements.txt not found in ${API_DIR}. Cannot install API dependencies."
      else
          VENV_DIR="${PROJECT_DIR}/venv"
          if [ -d "${VENV_DIR}" ]; then
              echo "Reusing existing virtual environment at ${VENV_DIR}."
          else
              echo "Creating Python virtual environment at ${VENV_DIR}..."
              "${PYTHON_BIN}" -m venv "${VENV_DIR}" || echo "Failed to create the virtual environment. Please check your Python installation."
          fi

          if [ -f "${VENV_DIR}/bin/activate" ]; then
              # shellcheck source=/dev/null
              source "${VENV_DIR}/bin/activate"
              echo "Virtual environment activated."

              echo "Installing API Python dependencies from requirements.txt..."
              pip3 install -r requirements.txt
              echo "API dependencies installed."

              echo "To activate this virtual environment later, run: source ${VENV_DIR}/bin/activate"
              echo "To start the API server (from ${PROJECT_DIR}, with the venv activated):"
              echo "  uvicorn code.backend.core.app:app --reload"
              deactivate
              echo "Virtual environment deactivated."
          fi
      fi
    )
fi

# --- Web Frontend Setup (React/Vite) ---
echo ""
echo "Setting up Quantis Web Frontend..."
WEB_FRONTEND_DIR="${PROJECT_DIR}/web-frontend"

if [ ! -d "${WEB_FRONTEND_DIR}" ]; then
    echo "Error: Web Frontend directory ${WEB_FRONTEND_DIR} not found. Skipping Web Frontend setup."
elif [ ! -f "${WEB_FRONTEND_DIR}/package.json" ]; then
    echo "Error: package.json not found in ${WEB_FRONTEND_DIR}. Cannot install Web Frontend dependencies."
else
    (
      cd "${WEB_FRONTEND_DIR}" || exit 1
      echo "Changed directory to $(pwd) for Web Frontend setup."

      if ! command -v npm &> /dev/null; then
          echo "npm command not found. Please install Node.js and npm, then re-run this script."
      else
          echo "Installing Web Frontend Node.js dependencies using npm..."
          npm install
          echo "Web Frontend dependencies installed."
          echo "To start the Web Frontend development server (from ${WEB_FRONTEND_DIR}): npm run dev"
          echo "To build the Web Frontend for production (from ${WEB_FRONTEND_DIR}): npm run build"
      fi
    )
fi

# --- Mobile Frontend Setup (Expo/React Native) ---
echo ""
echo "Setting up Quantis Mobile Frontend..."
MOBILE_FRONTEND_DIR="${PROJECT_DIR}/mobile-frontend"

if [ ! -d "${MOBILE_FRONTEND_DIR}" ]; then
    echo "Error: Mobile Frontend directory ${MOBILE_FRONTEND_DIR} not found. Skipping Mobile Frontend setup."
elif [ ! -f "${MOBILE_FRONTEND_DIR}/package.json" ]; then
    echo "Error: package.json not found in ${MOBILE_FRONTEND_DIR}. Cannot install Mobile Frontend dependencies."
else
    (
      cd "${MOBILE_FRONTEND_DIR}" || exit 1
      echo "Changed directory to $(pwd) for Mobile Frontend setup."

      if ! command -v npm &> /dev/null; then
          echo "npm command not found. Please install Node.js and npm, then re-run this script."
      else
          echo "Installing Mobile Frontend Node.js dependencies using npm..."
          npm install
          echo "Mobile Frontend dependencies installed."
          echo "To start the Mobile Frontend dev server (from ${MOBILE_FRONTEND_DIR}): npm start"
          echo "  Then press 'a' for Android, 'i' for iOS, or 'w' for web in the Expo CLI."
          echo "Native production builds are produced via EAS Build (https://docs.expo.dev/build/introduction/),"
          echo "not a local 'npm run build' - there is no such script for Expo apps."
      fi
    )
fi

# --- Quant ML library setup (Python) ---
echo ""
echo "Setting up Quantis quant_ml library..."
QUANT_ML_DIR="${PROJECT_DIR}/code/quant_ml"

if [ ! -d "${QUANT_ML_DIR}" ]; then
    echo "Warning: quant_ml directory ${QUANT_ML_DIR} not found. Skipping."
else
    if [ -f "${QUANT_ML_DIR}/requirements.txt" ]; then
        echo "Found a dedicated requirements.txt in ${QUANT_ML_DIR}."
        echo "Install it into the backend virtual environment with:"
        echo "  source ${PROJECT_DIR}/venv/bin/activate && pip3 install -r ${QUANT_ML_DIR}/requirements.txt"
    else
        echo "No dedicated requirements.txt in ${QUANT_ML_DIR}; it is imported directly by the backend"
        echo "and shares the backend's virtual environment and dependencies."
    fi
fi

# --- Docker Compose (Optional) ---
echo ""
INFRA_DIR="${PROJECT_DIR}/infrastructure"
if [ -f "${INFRA_DIR}/docker-compose.yml" ]; then
    echo "Found docker-compose.yml in ${INFRA_DIR}."
    echo "You can run the application using Docker Compose:"
    echo "  cd ${INFRA_DIR} && docker-compose up -d"
elif [ -f "${PROJECT_DIR}/docker-compose.yml" ]; then
    echo "Found docker-compose.yml in the project root ${PROJECT_DIR}."
    echo "You can run the application using Docker Compose:"
    echo "  cd ${PROJECT_DIR} && docker-compose up -d"
fi

echo ""
echo "Quantis project setup script finished."
echo "Please ensure all prerequisites (Python, Node.js, npm, Docker if used) are installed."
echo "Review the project's README.md and the instructions above for running each component."
