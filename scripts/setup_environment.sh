#!/bin/bash
# setup_environment.sh - Comprehensive environment setup script for Quantis project
#
# This script automates the setup of development environments for the Quantis project:
# - Installs all required dependencies
# - Sets up a virtual environment for the Python backend
# - Configures Node.js environments for the web and mobile frontends
# - Sets up database connection configuration
# - Configures monitoring tools
#
# Usage: ./setup_environment.sh [options]
# Options:
#   --all                Setup complete environment
#   --python             Setup only Python environment
#   --node               Setup only Node.js environment
#   --db                 Setup only database connections
#   --monitoring         Setup only monitoring tools
#   --dev                Setup for development (default)
#   --prod               Setup for production
#   --help               Show this help message
#
# Author: Abrar Ahmed
# Date: May 22, 2025

set -uo pipefail

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default settings
SETUP_PYTHON=false
SETUP_NODE=false
SETUP_DB=false
SETUP_MONITORING=false
SETUP_ALL=false
ENV="development"
# Resolve the actual repository root instead of trusting the caller's
# current directory - otherwise every path below silently resolves to the
# wrong place depending on where this script happens to be invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Function to display help message
show_help() {
    echo -e "${BLUE}Environment Setup Script for Quantis Project${NC}"
    echo ""
    echo "Usage: ./setup_environment.sh [options]"
    echo ""
    echo "Options:"
    echo "  --all                Setup complete environment"
    echo "  --python             Setup only Python environment"
    echo "  --node               Setup only Node.js environment"
    echo "  --db                 Setup only database connections"
    echo "  --monitoring         Setup only monitoring tools"
    echo "  --dev                Setup for development (default)"
    echo "  --prod               Setup for production"
    echo "  --help               Show this help message"
    echo ""
    exit 0
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_system_requirements() {
    echo -e "${BLUE}Checking system requirements...${NC}"

    OS=$(uname -s)
    echo "Operating System: $OS"

    if command_exists free; then
        TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
        echo "Total Memory: ${TOTAL_MEM}MB"

        if [ "$TOTAL_MEM" -lt 4000 ]; then
            echo -e "${YELLOW}Warning: Less than 4GB of RAM available. Performance may be affected.${NC}"
        fi
    fi

    DISK_SPACE=$(df -h "${PROJECT_ROOT}" | awk 'NR==2 {print $4}')
    echo "Available Disk Space: $DISK_SPACE"

    if command_exists nproc; then
        CPU_CORES=$(nproc)
        echo "CPU Cores: $CPU_CORES"
    fi

    echo -e "${GREEN}System requirements checked.${NC}"
}

# Function to setup the Python environment
setup_python_env() {
    echo -e "${BLUE}Setting up Python environment...${NC}"

    if ! command_exists python3; then
        echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
        echo "Please install Python 3 and try again."
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    echo "Python Version: $PYTHON_VERSION"

    API_DIR="${PROJECT_ROOT}/code/backend"
    if [ ! -d "${API_DIR}" ]; then
        echo -e "${YELLOW}Warning: Backend directory not found at ${API_DIR}.${NC}"
        return
    fi

    # A single shared virtual environment is used for the backend API and
    # the quant_ml library it imports (there is no separate top-level
    # "models" service with its own environment in this repository).
    VENV_DIR="${PROJECT_ROOT}/venv"
    echo "Creating/verifying the virtual environment..."
    if [ ! -d "${VENV_DIR}" ]; then
        echo "Creating virtual environment at ${VENV_DIR}..."
        python3 -m venv "${VENV_DIR}"
    else
        echo "Virtual environment already exists at ${VENV_DIR}."
    fi

    echo "Installing backend dependencies..."
    (
        set -e
        # shellcheck source=/dev/null
        source "${VENV_DIR}/bin/activate"
        pip install --upgrade pip --quiet

        if [ -f "${API_DIR}/requirements.txt" ]; then
            pip install -r "${API_DIR}/requirements.txt"
        else
            echo -e "${YELLOW}Warning: requirements.txt not found for the backend.${NC}"
        fi

        if [ "$ENV" = "development" ] && [ -f "${API_DIR}/tests/requirements-test.txt" ]; then
            pip install -r "${API_DIR}/tests/requirements-test.txt"
        fi

        deactivate
    )

    QUANT_ML_DIR="${PROJECT_ROOT}/code/quant_ml"
    if [ -d "${QUANT_ML_DIR}" ] && [ -f "${QUANT_ML_DIR}/requirements.txt" ]; then
        echo "Installing quant_ml-specific dependencies into the same virtual environment..."
        (
            set -e
            # shellcheck source=/dev/null
            source "${VENV_DIR}/bin/activate"
            pip install -r "${QUANT_ML_DIR}/requirements.txt"
            deactivate
        )
    fi

    echo -e "${GREEN}Python environment setup completed.${NC}"
}

# Function to setup Node.js environments (web + mobile frontends)
setup_node_env() {
    echo -e "${BLUE}Setting up Node.js environment...${NC}"

    if ! command_exists node; then
        echo -e "${RED}Error: Node.js is required but not installed.${NC}"
        echo "Please install Node.js and try again."
        exit 1
    fi
    NODE_VERSION=$(node --version)
    echo "Node.js Version: $NODE_VERSION"

    if ! command_exists npm; then
        echo -e "${RED}Error: npm is required but not installed.${NC}"
        echo "Please install npm and try again."
        exit 1
    fi
    NPM_VERSION=$(npm --version)
    echo "npm Version: $NPM_VERSION"

    for frontend in "web-frontend" "mobile-frontend"; do
        FRONTEND_DIR="${PROJECT_ROOT}/${frontend}"
        if [ ! -d "${FRONTEND_DIR}" ]; then
            echo -e "${YELLOW}Warning: ${frontend} directory not found.${NC}"
            continue
        fi

        (
            cd "${FRONTEND_DIR}" || exit 1
            echo "Installing ${frontend} dependencies..."
            npm install

            # Both frontends ship a single .env.example (not separate
            # per-environment example files), so that's what we seed from.
            if [ -f ".env.example" ] && [ ! -f ".env" ]; then
                echo "Creating ${frontend} environment configuration from .env.example..."
                cp .env.example .env
            fi
        )
    done

    echo -e "${GREEN}Node.js environment setup completed.${NC}"
}

# Function to set up database connection configuration
setup_db_connections() {
    echo -e "${BLUE}Setting up database connections...${NC}"

    if ! command_exists psql; then
        echo -e "${YELLOW}Warning: PostgreSQL client is not installed.${NC}"
        echo "Some database setup steps may be skipped."
    else
        echo "PostgreSQL client is installed."
    fi

    # The backend is configured via environment variables (see
    # code/backend/core/config.py and code/backend/.env.example) rather
    # than a database.yml file, so there is no such template to seed here.
    API_DIR="${PROJECT_ROOT}/code/backend"
    if [ -f "${API_DIR}/.env.example" ] && [ ! -f "${API_DIR}/.env" ]; then
        echo "Creating backend environment configuration from .env.example..."
        cp "${API_DIR}/.env.example" "${API_DIR}/.env"
    fi

    # Time-series / monitoring configuration lives under infrastructure/monitoring.
    MONITORING_DIR="${PROJECT_ROOT}/infrastructure/monitoring"
    if [ -d "${MONITORING_DIR}" ] && [ -f "${MONITORING_DIR}/influxdb.conf.example" ] && [ ! -f "${MONITORING_DIR}/influxdb.conf" ]; then
        echo "Creating InfluxDB configuration..."
        cp "${MONITORING_DIR}/influxdb.conf.example" "${MONITORING_DIR}/influxdb.conf"
    fi

    echo -e "${GREEN}Database connections setup completed.${NC}"
}

# Function to set up monitoring tools
setup_monitoring_tools() {
    echo -e "${BLUE}Setting up monitoring tools...${NC}"

    MONITORING_DIR="${PROJECT_ROOT}/infrastructure/monitoring"
    if [ ! -d "${MONITORING_DIR}" ]; then
        echo -e "${YELLOW}Warning: Monitoring directory not found at ${MONITORING_DIR}.${NC}"
        return
    fi

    if [ -f "${MONITORING_DIR}/prometheus.yml" ]; then
        echo "Prometheus configuration already present at ${MONITORING_DIR}/prometheus.yml."
    elif [ -f "${MONITORING_DIR}/prometheus.yml.example" ]; then
        echo "Creating Prometheus configuration..."
        cp "${MONITORING_DIR}/prometheus.yml.example" "${MONITORING_DIR}/prometheus.yml"
    fi

    if [ -d "${MONITORING_DIR}/grafana_dashboards" ]; then
        echo "Grafana dashboards found at ${MONITORING_DIR}/grafana_dashboards."
    fi

    if [ -f "${MONITORING_DIR}/alert_rules.yml" ]; then
        echo "Alerting rules already present at ${MONITORING_DIR}/alert_rules.yml."
    elif [ -f "${MONITORING_DIR}/alerting_rules.yml.example" ]; then
        echo "Creating alerting rules configuration..."
        cp "${MONITORING_DIR}/alerting_rules.yml.example" "${MONITORING_DIR}/alerting_rules.yml"
    fi

    echo -e "${GREEN}Monitoring tools setup completed.${NC}"
}

# Function to create a project-wide .env file
create_env_file() {
    echo -e "${BLUE}Creating environment configuration file...${NC}"

    ENV_FILE="$PROJECT_ROOT/.env"

    if [ ! -f "$ENV_FILE" ]; then
        echo "Creating .env file..."

        cat > "$ENV_FILE" << EOF
# Quantis Environment Configuration
# Generated by setup_environment.sh on $(date)
# Environment: $ENV

# API Configuration
API_PORT=8000
API_HOST=0.0.0.0
API_LOG_LEVEL=info

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=quantis_$ENV
DB_USER=quantis
DB_PASSWORD=quantis_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Frontend Configuration
FRONTEND_PORT=3000
API_BASE_URL=http://localhost:8000

# Monitoring Configuration
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001

# Feature Flags
ENABLE_FEATURE_X=true
ENABLE_FEATURE_Y=false
EOF

        echo -e "${GREEN}.env file created at $ENV_FILE${NC}"
    else
        echo -e "${YELLOW}Warning: .env file already exists. Skipping creation.${NC}"
    fi
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    show_help
fi

while [ $# -gt 0 ]; do
    case $1 in
        --all )     SETUP_PYTHON=true
                    SETUP_NODE=true
                    SETUP_DB=true
                    SETUP_MONITORING=true
                    SETUP_ALL=true
                    ;;
        --python )  SETUP_PYTHON=true
                    ;;
        --node )    SETUP_NODE=true
                    ;;
        --db )      SETUP_DB=true
                    ;;
        --monitoring ) SETUP_MONITORING=true
                    ;;
        --dev )     ENV="development"
                    ;;
        --prod )    ENV="production"
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
echo -e "${BLUE}Starting Quantis environment setup...${NC}"
echo -e "Environment: ${YELLOW}$ENV${NC}"
echo -e "Repository root: ${PROJECT_ROOT}"

check_system_requirements

if $SETUP_PYTHON || $SETUP_ALL; then
    setup_python_env
fi

if $SETUP_NODE || $SETUP_ALL; then
    setup_node_env
fi

if $SETUP_DB || $SETUP_ALL; then
    setup_db_connections
fi

if $SETUP_MONITORING || $SETUP_ALL; then
    setup_monitoring_tools
fi

if $SETUP_ALL; then
    create_env_file
fi

echo -e "${GREEN}Quantis environment setup completed successfully!${NC}"
echo -e "${YELLOW}Note: You may need to restart your terminal or run 'source .env' to apply environment variables.${NC}"
exit 0
