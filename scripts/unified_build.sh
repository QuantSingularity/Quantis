#!/bin/bash
# unified_build.sh - Comprehensive build script for Quantis project
#
# This script automates the build process for all components of the Quantis project:
# - API (Python backend, code/backend)
# - quant_ml (Python ML library imported by the backend, code/quant_ml)
# - Web Frontend (React/Vite, web-frontend)
# - Mobile Frontend (Expo/React Native, mobile-frontend)
#
# Usage: ./unified_build.sh [options]
# Options:
#   --all                Build all components
#   --api                Build only API
#   --models             Verify only the quant_ml library
#   --web                Build only web frontend
#   --mobile             Prepare only mobile frontend (install deps + typecheck)
#   --clean              Clean build artifacts before building
#   --prod               Build for production
#   --dev                Build for development (default)
#   --help               Show this help message
#
# Author: Abrar Ahmed
# Date: May 22, 2025

set -e  # Exit immediately if a command exits with a non-zero status

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default settings
BUILD_API=false
BUILD_MODELS=false
BUILD_WEB=false
BUILD_MOBILE=false
BUILD_ALL=false
CLEAN_BUILD=false
ENV="development"
# Resolve the actual repository root instead of trusting the caller's
# current directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Function to display help message
show_help() {
    echo -e "${BLUE}Unified Build Script for Quantis Project${NC}"
    echo ""
    echo "Usage: ./unified_build.sh [options]"
    echo ""
    echo "Options:"
    echo "  --all                Build all components"
    echo "  --api                Build only API"
    echo "  --models             Verify only the quant_ml library"
    echo "  --web                Build only web frontend"
    echo "  --mobile             Prepare only mobile frontend (install deps + typecheck)"
    echo "  --clean              Clean build artifacts before building"
    echo "  --prod               Build for production"
    echo "  --dev                Build for development (default)"
    echo "  --help               Show this help message"
    echo ""
    exit 0
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check required dependencies
check_dependencies() {
    echo -e "${BLUE}Checking dependencies...${NC}"

    # Check Python
    if ! command_exists python3; then
        echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
        exit 1
    fi

    # Check Node.js for frontend builds
    if ($BUILD_WEB || $BUILD_MOBILE) && ! command_exists node; then
        echo -e "${RED}Error: Node.js is required for frontend builds but not installed.${NC}"
        exit 1
    fi

    # Check npm for frontend builds
    if ($BUILD_WEB || $BUILD_MOBILE) && ! command_exists npm; then
        echo -e "${RED}Error: npm is required for frontend builds but not installed.${NC}"
        exit 1
    fi

    echo -e "${GREEN}All required dependencies are installed.${NC}"
}

# Function to clean build artifacts
clean_build() {
    echo -e "${BLUE}Cleaning build artifacts...${NC}"

    if $BUILD_API || $BUILD_ALL; then
        echo "Cleaning API build artifacts..."
        find "${PROJECT_ROOT}/code/backend" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find "${PROJECT_ROOT}/code/backend" -name "*.pyc" -delete
    fi

    if $BUILD_MODELS || $BUILD_ALL; then
        echo "Cleaning quant_ml build artifacts..."
        find "${PROJECT_ROOT}/code/quant_ml" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find "${PROJECT_ROOT}/code/quant_ml" -name "*.pyc" -delete
    fi

    if $BUILD_WEB || $BUILD_ALL; then
        echo "Cleaning web frontend build artifacts..."
        rm -rf "${PROJECT_ROOT}/web-frontend/build" "${PROJECT_ROOT}/web-frontend/node_modules/.cache"
    fi

    if $BUILD_MOBILE || $BUILD_ALL; then
        echo "Cleaning mobile frontend build artifacts..."
        rm -rf "${PROJECT_ROOT}/mobile-frontend/.expo" "${PROJECT_ROOT}/mobile-frontend/node_modules/.cache"
    fi

    echo -e "${GREEN}Build artifacts cleaned successfully.${NC}"
}

# Function to build API
build_api() {
    echo -e "${BLUE}Building API...${NC}"

    API_DIR="${PROJECT_ROOT}/code/backend"
    if [ ! -d "${API_DIR}" ]; then
        echo -e "${RED}Error: API directory not found at ${API_DIR}.${NC}"
        exit 1
    fi

    # The backend and quant_ml share a single virtual environment at the
    # repository root (there is no separate api/venv directory).
    VENV_DIR="${PROJECT_ROOT}/venv"
    if [ ! -d "${VENV_DIR}" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "${VENV_DIR}"
    fi

    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"

    echo "Installing API dependencies..."
    pip install -q -r "${API_DIR}/requirements.txt"

    echo "Verifying the API package imports cleanly..."
    (cd "${PROJECT_ROOT}" && PYTHONPATH="${PROJECT_ROOT}" python3 -c "from code.backend.core.app import app")

    deactivate

    echo -e "${GREEN}API built successfully.${NC}"
}

# Function to verify the quant_ml library
build_models() {
    echo -e "${BLUE}Verifying quant_ml library...${NC}"

    MODELS_DIR="${PROJECT_ROOT}/code/quant_ml"
    if [ ! -d "${MODELS_DIR}" ]; then
        echo -e "${RED}Error: quant_ml directory not found at ${MODELS_DIR}.${NC}"
        exit 1
    fi

    # quant_ml is a library imported by the backend, not a standalone
    # service — it shares the backend's virtual environment rather than
    # having its own venv/requirements.txt/setup.py.
    VENV_DIR="${PROJECT_ROOT}/venv"
    if [ ! -d "${VENV_DIR}" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "${VENV_DIR}"
    fi

    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"

    if [ -f "${MODELS_DIR}/requirements.txt" ]; then
        echo "Installing quant_ml-specific dependencies..."
        pip install -q -r "${MODELS_DIR}/requirements.txt"
    else
        # Ensure the backend's dependencies (which quant_ml relies on) are present.
        pip install -q -r "${PROJECT_ROOT}/code/backend/requirements.txt"
    fi

    echo "Verifying the quant_ml package imports cleanly..."
    (cd "${PROJECT_ROOT}" && PYTHONPATH="${PROJECT_ROOT}" python3 -c "import code.quant_ml")

    deactivate

    echo -e "${GREEN}quant_ml library verified successfully.${NC}"
}

# Function to build web frontend
build_web_frontend() {
    echo -e "${BLUE}Building web frontend...${NC}"

    WEB_DIR="${PROJECT_ROOT}/web-frontend"
    if [ ! -d "${WEB_DIR}" ]; then
        echo -e "${RED}Error: Web Frontend directory not found at ${WEB_DIR}.${NC}"
        exit 1
    fi

    (
      cd "${WEB_DIR}"
      echo "Installing web frontend dependencies..."
      npm install --no-audit --no-fund

      # The web frontend has a single Vite "build" script (no separate
      # build:dev variant) — Vite already picks the right mode/env file
      # based on NODE_ENV / --mode, so both dev and prod builds use it.
      echo "Building web frontend for $ENV environment..."
      if [ "$ENV" = "production" ]; then
        npm run build
      else
        npx vite build --mode development
      fi
    )

    echo -e "${GREEN}Web frontend built successfully.${NC}"
}

# Function to prepare the mobile frontend
build_mobile_frontend() {
    echo -e "${BLUE}Preparing mobile frontend...${NC}"

    MOBILE_DIR="${PROJECT_ROOT}/mobile-frontend"
    if [ ! -d "${MOBILE_DIR}" ]; then
        echo -e "${RED}Error: Mobile Frontend directory not found at ${MOBILE_DIR}.${NC}"
        exit 1
    fi

    (
      cd "${MOBILE_DIR}"
      echo "Installing mobile frontend dependencies..."
      npm install --no-audit --no-fund

      echo "Type-checking the mobile frontend..."
      npm run typecheck

      if [ "$ENV" = "production" ]; then
        echo "Exporting a static web bundle (npx expo export --platform web)..."
        npx expo export --platform web
        echo "Note: native iOS/Android binaries are produced via EAS Build, not"
        echo "by this script — see https://docs.expo.dev/build/introduction/"
      else
        echo "Mobile frontend is ready. Start it with: npm start"
      fi
    )

    echo -e "${GREEN}Mobile frontend prepared successfully.${NC}"
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    show_help
fi

while [ "$1" != "" ]; do
    case $1 in
        --all )     BUILD_API=true
                    BUILD_MODELS=true
                    BUILD_WEB=true
                    BUILD_MOBILE=true
                    BUILD_ALL=true
                    ;;
        --api )     BUILD_API=true
                    ;;
        --models )  BUILD_MODELS=true
                    ;;
        --web )     BUILD_WEB=true
                    ;;
        --mobile )  BUILD_MOBILE=true
                    ;;
        --clean )   CLEAN_BUILD=true
                    ;;
        --prod )    ENV="production"
                    ;;
        --dev )     ENV="development"
                    ;;
        --help )    show_help
                    ;;
        * )         echo -e "${RED}Error: Unknown option $1${NC}"
                    show_help
                    ;;
    esac
    shift
done

# Main execution
echo -e "${BLUE}Starting Quantis unified build process...${NC}"
echo -e "Environment: ${YELLOW}$ENV${NC}"
echo -e "Repository root: ${PROJECT_ROOT}"

# Check dependencies
check_dependencies

# Clean build artifacts if requested
if $CLEAN_BUILD; then
    clean_build
fi

# Build components
if $BUILD_API; then
    build_api
fi

if $BUILD_MODELS; then
    build_models
fi

if $BUILD_WEB; then
    build_web_frontend
fi

if $BUILD_MOBILE; then
    build_mobile_frontend
fi

echo -e "${GREEN}Quantis build process completed successfully!${NC}"
exit 0
