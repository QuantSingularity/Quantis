#!/bin/bash
# documentation_generator.sh - Automated documentation generator for Quantis project
#
# This script automates the generation of documentation for the Quantis project:
# - API documentation using Swagger/OpenAPI
# - Code documentation using appropriate tools (JSDoc, Sphinx)
# - README files for components
# - User guides and tutorials
#
# Usage: ./documentation_generator.sh [options]
# Options:
#   --all                Generate all documentation
#   --api                Generate only API documentation
#   --code               Generate only code documentation
#   --readme             Generate only README files
#   --guides             Generate only user guides
#   --output DIR         Output directory for documentation (default: ./docs)
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
GEN_API=false
GEN_CODE=false
GEN_README=false
GEN_GUIDES=false
GEN_ALL=false
# Resolve the actual repository root instead of trusting the caller's
# current directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/docs"

# Function to display help message
show_help() {
    echo -e "${BLUE}Documentation Generator for Quantis Project${NC}"
    echo ""
    echo "Usage: ./documentation_generator.sh [options]"
    echo ""
    echo "Options:"
    echo "  --all                Generate all documentation"
    echo "  --api                Generate only API documentation"
    echo "  --code               Generate only code documentation"
    echo "  --readme             Generate only README files"
    echo "  --guides             Generate only user guides"
    echo "  --output DIR         Output directory for documentation (default: ./docs)"
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
    echo -e "${BLUE}Checking documentation dependencies...${NC}"

    if $GEN_API || $GEN_ALL; then
        if ! command_exists swagger-cli; then
            echo -e "${YELLOW}Warning: swagger-cli is not installed. Installing...${NC}"
            npm install -g swagger-cli
        fi
    fi

    if $GEN_CODE || $GEN_ALL; then
        if [ -d "$PROJECT_ROOT/web-frontend" ] || [ -d "$PROJECT_ROOT/mobile-frontend" ]; then
            if ! command_exists jsdoc; then
                echo -e "${YELLOW}Warning: jsdoc is not installed. Installing...${NC}"
                npm install -g jsdoc
            fi
        fi

        if [ -d "$PROJECT_ROOT/code/backend" ] || [ -d "$PROJECT_ROOT/code/quant_ml" ]; then
            if ! command_exists sphinx-build; then
                echo -e "${YELLOW}Warning: sphinx is not installed. Installing...${NC}"
                pip install sphinx sphinx_rtd_theme
            fi
        fi
    fi

    if $GEN_README || $GEN_GUIDES || $GEN_ALL; then
        if ! command_exists pandoc; then
            echo -e "${YELLOW}Warning: pandoc is not installed. Installing...${NC}"
            if command_exists apt-get; then
                if sudo -n true 2>/dev/null; then
                    sudo apt-get update && sudo apt-get install -y pandoc
                else
                    echo -e "${RED}Error: pandoc is missing and passwordless sudo is not available.${NC}"
                    echo -e "${RED}Install it manually with: sudo apt-get install -y pandoc${NC}"
                    exit 1
                fi
            elif command_exists brew; then
                brew install pandoc
            else
                echo -e "${RED}Error: Cannot install pandoc. Please install it manually.${NC}"
                exit 1
            fi
        fi
    fi

    echo -e "${GREEN}All required documentation dependencies are installed.${NC}"
}

# Function to prepare output directory
prepare_output_dir() {
    echo -e "${BLUE}Preparing output directory...${NC}"

    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/api"
    mkdir -p "$OUTPUT_DIR/code"
    mkdir -p "$OUTPUT_DIR/readme"
    mkdir -p "$OUTPUT_DIR/guides"

    echo -e "${GREEN}Output directory prepared: $OUTPUT_DIR${NC}"
}

# Function to generate API documentation
generate_api_docs() {
    echo -e "${BLUE}Generating API documentation...${NC}"

    if [ -d "$PROJECT_ROOT/code/backend" ]; then
        cd "$PROJECT_ROOT/code/backend"

        # Look for OpenAPI/Swagger files
        SWAGGER_FILES=$(find . -name "swagger.yaml" -o -name "swagger.json" -o -name "openapi.yaml" -o -name "openapi.json")

        if [ -z "$SWAGGER_FILES" ]; then
            echo -e "${YELLOW}Warning: No OpenAPI/Swagger files found. Generating from code...${NC}"

            # Check if FastAPI is used (can auto-generate OpenAPI docs)
            if grep -q "fastapi" requirements.txt 2>/dev/null; then
                echo "Detected FastAPI. Generating OpenAPI schema..."

                # Create a temporary script to extract OpenAPI schema
                cat > extract_openapi.py << EOF
import json
import sys

sys.path.insert(0, "$PROJECT_ROOT")

from fastapi.openapi.utils import get_openapi

# The FastAPI app instance lives at code.backend.core.app, not a flat
# app.py/main.py/api.py in this directory.
try:
    from code.backend.core.app import app
except Exception as e:
    print(f"Error importing code.backend.core.app: {e}")
    sys.exit(1)

openapi_schema = get_openapi(
    title="Quantis API",
    version="1.0.0",
    description="Quantis API Documentation",
    routes=app.routes,
)

with open("openapi.json", "w") as f:
    json.dump(openapi_schema, f, indent=2)

print("OpenAPI schema generated successfully.")
EOF

                # Run the script to extract OpenAPI schema
                PYTHONPATH="$PROJECT_ROOT" python3 extract_openapi.py
                rm extract_openapi.py

                SWAGGER_FILES="openapi.json"
            fi
        fi

        # Generate HTML documentation from OpenAPI/Swagger files
        if [ -n "$SWAGGER_FILES" ]; then
            for file in $SWAGGER_FILES; do
                echo "Generating HTML documentation from $file..."

                # Use swagger-cli to validate
                swagger-cli validate "$file"

                # Use Redoc to generate HTML documentation
                npx redoc-cli bundle "$file" -o "$OUTPUT_DIR/api/index.html"
            done
        else
            echo -e "${YELLOW}Warning: Could not generate API documentation. No OpenAPI/Swagger files found.${NC}"
        fi
    else
        echo -e "${YELLOW}Warning: API directory not found.${NC}"
    fi

    echo -e "${GREEN}API documentation generation completed.${NC}"
}

# Function to generate code documentation
generate_code_docs() {
    echo -e "${BLUE}Generating code documentation...${NC}"

    # Generate JavaScript/TypeScript documentation using JSDoc
    if [ -d "$PROJECT_ROOT/web-frontend" ]; then
        echo "Generating web frontend code documentation..."
        cd "$PROJECT_ROOT/web-frontend"

        # Create JSDoc configuration if it doesn't exist
        JSDOC_CONFIG_CREATED=false
        if [ ! -f "jsdoc.json" ]; then
            JSDOC_CONFIG_CREATED=true
            cat > jsdoc.json << EOF
{
  "source": {
    "include": ["src"],
    "includePattern": ".+\\.(js|jsx|ts|tsx)$",
    "excludePattern": "(node_modules|docs)"
  },
  "plugins": ["plugins/markdown"],
  "opts": {
    "destination": "$OUTPUT_DIR/code/web-frontend",
    "recurse": true,
    "readme": "README.md"
  }
}
EOF
        fi

        # Run JSDoc
        jsdoc -c jsdoc.json
        if $JSDOC_CONFIG_CREATED; then
            rm -f jsdoc.json
        fi
    fi

    if [ -d "$PROJECT_ROOT/mobile-frontend" ]; then
        echo "Generating mobile frontend code documentation..."
        cd "$PROJECT_ROOT/mobile-frontend"

        # Create JSDoc configuration if it doesn't exist
        JSDOC_CONFIG_CREATED=false
        if [ ! -f "jsdoc.json" ]; then
            JSDOC_CONFIG_CREATED=true
            cat > jsdoc.json << EOF
{
  "source": {
    "include": ["src"],
    "includePattern": ".+\\.(js|jsx|ts|tsx)$",
    "excludePattern": "(node_modules|docs)"
  },
  "plugins": ["plugins/markdown"],
  "opts": {
    "destination": "$OUTPUT_DIR/code/mobile-frontend",
    "recurse": true,
    "readme": "README.md"
  }
}
EOF
        fi

        # Run JSDoc
        jsdoc -c jsdoc.json
        if $JSDOC_CONFIG_CREATED; then
            rm -f jsdoc.json
        fi
    fi

    # Generate Python documentation using Sphinx
    if [ -d "$PROJECT_ROOT/code/backend" ] || [ -d "$PROJECT_ROOT/code/quant_ml" ]; then
        echo "Generating Python code documentation..."

        # Create Sphinx documentation directory
        mkdir -p "$OUTPUT_DIR/code/python"
        cd "$OUTPUT_DIR/code/python"

        # Initialize Sphinx
        sphinx-quickstart -q -p "Quantis" -a "Quantis Team" -v "1.0" --ext-autodoc --ext-viewcode --ext-todo

        # Update conf.py to include autodoc
        sed -i "s/extensions = \[/extensions = \['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.todo',/g" source/conf.py

        # Create modules.rst
        cat > source/modules.rst << EOF
API Modules
===========

.. toctree::
   :maxdepth: 4

   backend
   quant_ml
EOF

        # Generate backend module documentation
        if [ -d "$PROJECT_ROOT/code/backend" ]; then
            sphinx-apidoc -o source/backend "$PROJECT_ROOT/code/backend" -H "Backend API" -M -e -f
        fi

        # Generate quant_ml module documentation
        if [ -d "$PROJECT_ROOT/code/quant_ml" ]; then
            sphinx-apidoc -o source/quant_ml "$PROJECT_ROOT/code/quant_ml" -H "Quant ML" -M -e -f
        fi

        # Build HTML documentation
        make html
    fi

    echo -e "${GREEN}Code documentation generation completed.${NC}"
}

# Function to generate README files
generate_readme_files() {
    echo -e "${BLUE}Generating README files...${NC}"

    # Create main README if it doesn't exist
    if [ ! -f "$PROJECT_ROOT/README.md" ]; then
        echo "Creating main README.md..."

        cat > "$PROJECT_ROOT/README.md" << EOF
# Quantis

## Quantitative Trading & Investment Analytics Platform

Quantis is a comprehensive quantitative trading and investment analytics platform that combines advanced statistical models, machine learning algorithms, and real-time market data to provide powerful insights and automated trading strategies.

## Overview

Quantis provides a robust platform for quantitative analysis, algorithmic trading, and investment portfolio optimization. The system leverages advanced statistical models and machine learning algorithms to analyze market data, identify trading opportunities, and execute automated trading strategies.

## Key Features

### Data Processing & Analysis

- **Real-time Market Data**: Integration with multiple data sources for comprehensive market coverage
- **Historical Data Analysis**: Tools for backtesting and historical performance evaluation
- **Alternative Data Processing**: Analysis of non-traditional data sources for unique insights
- **Data Quality Assurance**: Automated validation and cleaning of financial data

### Quantitative Analysis

- **Statistical Models**: Time series analysis, regression models, and factor analysis
- **Machine Learning Algorithms**: Classification, regression, and clustering for market prediction
- **Risk Models**: Value at Risk (VaR), Expected Shortfall, and stress testing
- **Portfolio Optimization**: Modern Portfolio Theory implementation with custom constraints

### Trading Strategies

- **Strategy Development Framework**: Tools for creating and testing trading algorithms
- **Algorithmic Trading**: Automated execution of trading strategies
- **Signal Generation**: Technical and fundamental indicators for trade signals
- **Backtesting Engine**: Historical performance evaluation of trading strategies

### Portfolio Management

- **Asset Allocation**: Optimal distribution of investments across asset classes
- **Risk Management**: Tools for monitoring and controlling portfolio risk
- **Performance Analytics**: Comprehensive metrics for evaluating investment performance
- **Rebalancing Strategies**: Automated portfolio rebalancing based on defined rules

## Technology Stack

### Backend

- **Language**: Python, Rust (for performance-critical components)
- **Framework**: FastAPI, Flask
- **Database**: PostgreSQL, InfluxDB (time series data)
- **Task Queue**: Celery, Redis
- **ML Libraries**: scikit-learn, PyTorch, TensorFlow
- **Financial Libraries**: pandas-ta, pyfolio, zipline

### Frontend

- **Framework**: React with TypeScript
- **State Management**: Redux Toolkit
- **Data Visualization**: D3.js, Plotly, TradingView
- **Styling**: Tailwind CSS, Styled Components
- **API Client**: Axios, React Query

### Infrastructure

- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus, Grafana
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)

## Getting Started

1. Clone the repository
2. Run \`./setup_quantis_env.sh\` to set up the development environment
3. Run \`./run_quantis.sh\` to start the application

## Documentation

For detailed documentation, see the \`docs\` directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
EOF
    fi

    # Copy main README to output directory
    cp "$PROJECT_ROOT/README.md" "$OUTPUT_DIR/readme/"

    # Generate component-specific READMEs. Component labels don't match
    # directory names 1:1 in this repository (the API lives at
    # code/backend, models at code/quant_ml, monitoring config at
    # infrastructure/monitoring), so map each label to its real path.
    for component in api models web-frontend mobile-frontend infrastructure monitoring; do
        case $component in
            api)            component_dir="$PROJECT_ROOT/code/backend" ;;
            models)         component_dir="$PROJECT_ROOT/code/quant_ml" ;;
            monitoring)     component_dir="$PROJECT_ROOT/infrastructure/monitoring" ;;
            *)              component_dir="$PROJECT_ROOT/$component" ;;
        esac

        if [ -d "$component_dir" ] && [ ! -f "$component_dir/README.md" ]; then
            echo "Creating README.md for $component..."

            case $component in
                api)
                    cat > "$component_dir/README.md" << EOF
# Quantis API

This directory contains the backend API for the Quantis platform.

## Overview

The API provides endpoints for data retrieval, model execution, and trading operations.

## Getting Started

1. Create a virtual environment: \`python3 -m venv venv\` (from the repository root)
2. Activate the virtual environment: \`source venv/bin/activate\`
3. Install dependencies: \`pip install -r code/backend/requirements.txt\`
4. Run the API (from the repository root): \`uvicorn code.backend.core.app:app --reload\`

## API Documentation

Interactive API documentation is available at \`/docs\` when the API is running.

## Directory Structure

- \`core/app.py\`: Main FastAPI application instance
- \`endpoints/\`: API route definitions
- \`domain/\`: Data models and schemas
- \`services/\`: Business logic
- \`auth/\`, \`middleware/\`: Authentication and request middleware
EOF
                    ;;
                models)
                    cat > "$component_dir/README.md" << EOF
# Quantis quant_ml

This directory contains the machine learning library used by the Quantis API.

## Overview

quant_ml provides predictive analytics for trading strategies and portfolio optimization,
and is imported directly by the backend (code/backend) — it is not a standalone service
and shares the backend's virtual environment and dependencies.

## Getting Started

From the repository root, with the backend virtual environment activated:

\`\`\`bash
source venv/bin/activate
python3 -c "import code.quant_ml"
\`\`\`
EOF
                    ;;
                web-frontend)
                    cat > "$component_dir/README.md" << EOF
# Quantis Web Frontend

This directory contains the web frontend for the Quantis platform (React + Vite + MUI).

## Overview

The web frontend provides a user interface for interacting with the Quantis platform.

## Getting Started

1. Install dependencies: \`npm install\`
2. Run the development server: \`npm run dev\`
3. Build for production: \`npm run build\`

## Directory Structure

- \`src/\`: Source code
  - \`components/\`: Shared UI components
  - \`pages/\`: Route-level page components
  - \`api/\`: API client modules
  - \`context/\`: React context providers (auth, theme)
  - \`theme/\`: Design tokens and MUI theme
- \`public/\`: Static assets
EOF
                    ;;
                mobile-frontend)
                    cat > "$component_dir/README.md" << EOF
# Quantis Mobile Frontend

This directory contains the mobile frontend for the Quantis platform (Expo + React Native).

## Overview

The mobile frontend provides a mobile user interface for interacting with the Quantis platform.

## Getting Started

1. Install dependencies: \`npm install\`
2. Run the development server: \`npm start\` (then press \`a\`/\`i\`/\`w\` in the Expo CLI)

Expo apps do not have a local \`npm run build\` step. Native binaries are produced via
EAS Build (\`npx eas-cli build --platform all\`); a static web export can be produced with
\`npx expo export --platform web\`.

## Directory Structure

- \`src/\`: Source code
  - \`components/\`: Shared UI components
  - \`screens/\`: Screen components
  - \`navigation/\`: React Navigation stacks/tabs
  - \`api/\`: API client modules
  - \`context/\`: React context providers (auth, theme)
- \`assets/\`: Static assets
EOF
                    ;;
                infrastructure)
                    cat > "$component_dir/README.md" << EOF
# Quantis Infrastructure

This directory contains the infrastructure configuration for the Quantis platform.

## Overview

The infrastructure configuration defines the deployment and operation of the Quantis platform.

## Getting Started

1. Initialize Terraform: \`cd terraform && terraform init\`
2. Plan deployment: \`terraform plan\`
3. Apply deployment: \`terraform apply\`

## Directory Structure

- \`terraform/\`: Terraform configuration
- \`kubernetes/\`: Kubernetes manifests
- \`ansible/\`: Ansible playbooks
- \`monitoring/\`: Prometheus/Grafana configuration
- \`mysql/\`: Database configuration
EOF
                    ;;
                monitoring)
                    cat > "$component_dir/README.md" << EOF
# Quantis Monitoring

This directory contains the monitoring configuration for the Quantis platform.

## Overview

The monitoring configuration defines the metrics collection, visualization, and alerting for the Quantis platform.

## Getting Started

1. Start the monitoring stack: \`docker compose up -d\` (or \`docker-compose up -d\` on older installs)
2. Access Prometheus: \`http://localhost:9090\`
3. Access Grafana: \`http://localhost:3000\`

## Directory Structure

- \`prometheus.yml\`: Prometheus configuration
- \`alert_rules.yml\`: Alerting rules
- \`alertmanager.yml\`: Alertmanager configuration
- \`grafana_dashboards/\`: Grafana dashboard definitions
- \`grafana_provisioning/\`: Grafana datasource/dashboard provisioning
EOF
                    ;;
            esac

            # Copy component README to output directory
            cp "$component_dir/README.md" "$OUTPUT_DIR/readme/$component.md"
        elif [ -f "$component_dir/README.md" ]; then
            # Copy existing component README to output directory
            cp "$component_dir/README.md" "$OUTPUT_DIR/readme/$component.md"
        fi
    done

    # Generate index file for READMEs
    cat > "$OUTPUT_DIR/readme/index.md" << EOF
# Quantis Documentation

## README Files

- [Main README](README.md)
EOF

    for component in api models web-frontend mobile-frontend infrastructure monitoring; do
        if [ -f "$OUTPUT_DIR/readme/$component.md" ]; then
            echo "- [$component]($component.md)" >> "$OUTPUT_DIR/readme/index.md"
        fi
    done

    echo -e "${GREEN}README files generation completed.${NC}"
}

# Function to generate user guides
generate_user_guides() {
    echo -e "${BLUE}Generating user guides...${NC}"

    # Create user guides directory
    mkdir -p "$OUTPUT_DIR/guides"

    # Generate installation guide
    cat > "$OUTPUT_DIR/guides/installation_guide.md" << EOF
# Quantis Installation Guide

This guide provides instructions for installing and setting up the Quantis platform.

## Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- Docker and Docker Compose
- PostgreSQL 13 or higher
- Redis

## Installation Steps

### 1. Clone the Repository

\`\`\`bash
git clone https://github.com/yourusername/quantis.git
cd quantis
\`\`\`

### 2. Set Up Environment

Run the environment setup script:

\`\`\`bash
./setup_quantis_env.sh --all
\`\`\`

This script will:
- Create Python virtual environments
- Install Python dependencies
- Install Node.js dependencies
- Set up database connections
- Configure monitoring tools

### 3. Start the Application

Run the application startup script:

\`\`\`bash
./run_quantis.sh
\`\`\`

This script will:
- Start the API server (which imports the quant_ml library directly — there
  is no separate standalone model service)
- Start the web frontend
- Wait for the API to become healthy before reporting success

### 4. Access the Application

- Web Frontend: http://localhost:3000
- API Documentation: http://localhost:8000/docs
- Monitoring Dashboard: http://localhost:3001

## Troubleshooting

### Common Issues

#### API Server Won't Start

Check the following:
- Ensure PostgreSQL is running
- Verify database connection settings in \`.env\`
- Check API logs for specific errors

#### quant_ml Import Errors

Check the following:
- Ensure Python dependencies are installed in the shared virtual environment
- Verify the \`code/quant_ml\` package imports cleanly: \`python3 -c "import code.quant_ml"\`
- Check the API server logs — quant_ml is imported directly by the backend,
  so import errors surface there rather than in a separate service log

#### Frontend Won't Load

Check the following:
- Ensure API server is running
- Verify API URL in frontend configuration
- Check browser console for errors

## Next Steps

After installation, refer to the [User Guide](user_guide.md) for information on using the Quantis platform.
EOF

    # Generate user guide
    cat > "$OUTPUT_DIR/guides/user_guide.md" << EOF
# Quantis User Guide

This guide provides instructions for using the Quantis platform.

## Getting Started

### Logging In

1. Navigate to the Quantis web application
2. Enter your username and password
3. Click "Log In"

### Dashboard Overview

The dashboard provides an overview of your portfolio and trading strategies:

- **Portfolio Summary**: Current portfolio value, allocation, and performance
- **Strategy Performance**: Performance metrics for active trading strategies
- **Market Overview**: Key market indicators and trends
- **Recent Alerts**: Recent alerts and notifications

## Portfolio Management

### Viewing Portfolio

1. Click "Portfolio" in the main navigation
2. View portfolio summary, allocation, and performance
3. Click on individual assets for detailed information

### Creating a Portfolio

1. Click "Portfolio" in the main navigation
2. Click "Create Portfolio"
3. Enter portfolio name and description
4. Set initial capital and risk parameters
5. Click "Create"

### Managing Portfolio

1. Click "Portfolio" in the main navigation
2. Select a portfolio from the list
3. Click "Manage"
4. Add or remove assets, adjust allocations, and set constraints
5. Click "Save Changes"

## Trading Strategies

### Viewing Strategies

1. Click "Strategies" in the main navigation
2. View strategy list, performance, and status
3. Click on individual strategies for detailed information

### Creating a Strategy

1. Click "Strategies" in the main navigation
2. Click "Create Strategy"
3. Select strategy type (e.g., momentum, mean reversion)
4. Configure strategy parameters
5. Set execution parameters (e.g., frequency, position sizing)
6. Click "Create"

### Backtesting a Strategy

1. Click "Strategies" in the main navigation
2. Select a strategy from the list
3. Click "Backtest"
4. Set backtest parameters (e.g., time period, initial capital)
5. Click "Run Backtest"
6. View backtest results and performance metrics

### Deploying a Strategy

1. Click "Strategies" in the main navigation
2. Select a strategy from the list
3. Click "Deploy"
4. Set deployment parameters (e.g., capital allocation, risk limits)
5. Click "Deploy Strategy"

## Data Analysis

### Viewing Market Data

1. Click "Data" in the main navigation
2. Select data type (e.g., price, volume, fundamentals)
3. Select instruments and time period
4. View data visualization

### Creating Custom Indicators

1. Click "Data" in the main navigation
2. Click "Custom Indicators"
3. Click "Create Indicator"
4. Define indicator formula and parameters
5. Click "Create"

### Running Analysis

1. Click "Data" in the main navigation
2. Click "Analysis"
3. Select analysis type (e.g., correlation, regression)
4. Configure analysis parameters
5. Click "Run Analysis"
6. View analysis results

## Monitoring and Alerts

### Viewing Monitoring Dashboard

1. Click "Monitoring" in the main navigation
2. View system status, performance metrics, and alerts
3. Click on individual metrics for detailed information

### Creating Alerts

1. Click "Monitoring" in the main navigation
2. Click "Alerts"
3. Click "Create Alert"
4. Select alert type and conditions
5. Set notification preferences
6. Click "Create"

## Account Management

### Updating Profile

1. Click your username in the top-right corner
2. Click "Profile"
3. Update profile information
4. Click "Save Changes"

### Managing API Keys

1. Click your username in the top-right corner
2. Click "API Keys"
3. View existing API keys or create new ones
4. Set permissions and expiration
5. Click "Create Key"

## Support and Feedback

### Getting Help

1. Click "Help" in the main navigation
2. Browse help topics or search for specific information
3. Contact support if needed

### Providing Feedback

1. Click "Help" in the main navigation
2. Click "Feedback"
3. Enter feedback and click "Submit"
EOF

    # Generate API guide
    cat > "$OUTPUT_DIR/guides/api_guide.md" << EOF
# Quantis API Guide

This guide provides instructions for using the Quantis API.

## Authentication

The Quantis API uses API keys for authentication. To obtain an API key:

1. Log in to the Quantis web application
2. Click your username in the top-right corner
3. Click "API Keys"
4. Click "Create API Key"
5. Set permissions and expiration
6. Click "Create Key"
7. Copy the API key (it will only be shown once)

Include the API key in the \`Authorization\` header of your requests:

\`\`\`
Authorization: Bearer YOUR_API_KEY
\`\`\`

## API Endpoints

### Portfolio Endpoints

#### Get Portfolios

\`\`\`
GET /api/portfolios
\`\`\`

Returns a list of portfolios.

#### Get Portfolio

\`\`\`
GET /api/portfolios/{portfolio_id}
\`\`\`

Returns details for a specific portfolio.

#### Create Portfolio

\`\`\`
POST /api/portfolios
\`\`\`

Creates a new portfolio.

Request body:
\`\`\`json
{
  "name": "My Portfolio",
  "description": "My investment portfolio",
  "initial_capital": 100000,
  "risk_parameters": {
    "max_drawdown": 0.1,
    "max_volatility": 0.2
  }
}
\`\`\`

#### Update Portfolio

\`\`\`
PUT /api/portfolios/{portfolio_id}
\`\`\`

Updates an existing portfolio.

#### Delete Portfolio

\`\`\`
DELETE /api/portfolios/{portfolio_id}
\`\`\`

Deletes a portfolio.

### Strategy Endpoints

#### Get Strategies

\`\`\`
GET /api/strategies
\`\`\`

Returns a list of strategies.

#### Get Strategy

\`\`\`
GET /api/strategies/{strategy_id}
\`\`\`

Returns details for a specific strategy.

#### Create Strategy

\`\`\`
POST /api/strategies
\`\`\`

Creates a new strategy.

Request body:
\`\`\`json
{
  "name": "My Strategy",
  "description": "My trading strategy",
  "type": "momentum",
  "parameters": {
    "lookback_period": 20,
    "threshold": 0.05
  },
  "execution_parameters": {
    "frequency": "daily",
    "position_sizing": "equal_weight"
  }
}
\`\`\`

#### Update Strategy

\`\`\`
PUT /api/strategies/{strategy_id}
\`\`\`

Updates an existing strategy.

#### Delete Strategy

\`\`\`
DELETE /api/strategies/{strategy_id}
\`\`\`

Deletes a strategy.

#### Backtest Strategy

\`\`\`
POST /api/strategies/{strategy_id}/backtest
\`\`\`

Runs a backtest for a strategy.

Request body:
\`\`\`json
{
  "start_date": "2020-01-01",
  "end_date": "2020-12-31",
  "initial_capital": 100000
}
\`\`\`

#### Deploy Strategy

\`\`\`
POST /api/strategies/{strategy_id}/deploy
\`\`\`

Deploys a strategy.

Request body:
\`\`\`json
{
  "capital_allocation": 100000,
  "risk_limits": {
    "max_drawdown": 0.1,
    "max_position_size": 0.1
  }
}
\`\`\`

### Data Endpoints

#### Get Market Data

\`\`\`
GET /api/data/market
\`\`\`

Returns market data.

Query parameters:
- \`instruments\`: Comma-separated list of instrument identifiers
- \`start_date\`: Start date (YYYY-MM-DD)
- \`end_date\`: End date (YYYY-MM-DD)
- \`interval\`: Data interval (e.g., 1d, 1h, 5m)

#### Get Indicators

\`\`\`
GET /api/data/indicators
\`\`\`

Returns indicator data.

Query parameters:
- \`instruments\`: Comma-separated list of instrument identifiers
- \`indicators\`: Comma-separated list of indicator identifiers
- \`start_date\`: Start date (YYYY-MM-DD)
- \`end_date\`: End date (YYYY-MM-DD)
- \`interval\`: Data interval (e.g., 1d, 1h, 5m)

## Error Handling

The API returns standard HTTP status codes:

- 200: Success
- 400: Bad request
- 401: Unauthorized
- 403: Forbidden
- 404: Not found
- 500: Internal server error

Error responses include a JSON body with error details:

\`\`\`json
{
  "error": {
    "code": "invalid_request",
    "message": "Invalid request parameters",
    "details": {
      "field": "start_date",
      "issue": "Invalid date format"
    }
  }
}
\`\`\`

## Rate Limiting

The API enforces rate limits to prevent abuse. Rate limit information is included in the response headers:

- \`X-RateLimit-Limit\`: Maximum number of requests allowed in the current period
- \`X-RateLimit-Remaining\`: Number of requests remaining in the current period
- \`X-RateLimit-Reset\`: Time when the rate limit will reset (Unix timestamp)

If you exceed the rate limit, you will receive a 429 Too Many Requests response.
EOF

    # Generate index file for guides
    cat > "$OUTPUT_DIR/guides/index.md" << EOF
# Quantis User Guides

## Available Guides

- [Installation Guide](installation_guide.md)
- [User Guide](user_guide.md)
- [API Guide](api_guide.md)

## Additional Resources

- [API Documentation](../api/index.html)
- [Code Documentation](../code/index.html)
EOF

    # Convert Markdown to HTML
    if command_exists pandoc; then
        echo "Converting Markdown guides to HTML..."

        for guide in "$OUTPUT_DIR/guides"/*.md; do
            basename=$(basename "$guide" .md)
            pandoc "$guide" -o "$OUTPUT_DIR/guides/$basename.html" --standalone --metadata title="Quantis - $(echo "$basename" | sed 's/_/ /g' | awk '{for(i=1;i<=NF;i++) $i=toupper(substr($i,1,1)) substr($i,2)} 1')"
        done
    fi

    echo -e "${GREEN}User guides generation completed.${NC}"
}

# Function to generate index file
generate_index() {
    echo -e "${BLUE}Generating index file...${NC}"

    cat > "$OUTPUT_DIR/index.html" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantis Documentation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        h1, h2, h3 {
            color: #0066cc;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .section {
            margin-bottom: 30px;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
        }
        .section-title {
            margin-top: 0;
        }
        .section-content {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .card {
            flex: 1 1 300px;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            background-color: #f9f9f9;
        }
        .card h3 {
            margin-top: 0;
        }
        a {
            color: #0066cc;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .footer {
            margin-top: 50px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Quantis Documentation</h1>
        <p>Welcome to the Quantis documentation. This page provides links to various documentation resources for the Quantis platform.</p>

        <div class="section">
            <h2 class="section-title">API Documentation</h2>
            <div class="section-content">
                <div class="card">
                    <h3>API Reference</h3>
                    <p>Comprehensive reference for the Quantis API endpoints, parameters, and responses.</p>
                    <p><a href="api/index.html">View API Documentation</a></p>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">Code Documentation</h2>
            <div class="section-content">
                <div class="card">
                    <h3>Python Code</h3>
                    <p>Documentation for the Python backend code, including API and models.</p>
                    <p><a href="code/python/build/html/index.html">View Python Documentation</a></p>
                </div>
                <div class="card">
                    <h3>Web Frontend</h3>
                    <p>Documentation for the web frontend code.</p>
                    <p><a href="code/web-frontend/index.html">View Web Frontend Documentation</a></p>
                </div>
                <div class="card">
                    <h3>Mobile Frontend</h3>
                    <p>Documentation for the mobile frontend code.</p>
                    <p><a href="code/mobile-frontend/index.html">View Mobile Frontend Documentation</a></p>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">README Files</h2>
            <div class="section-content">
                <div class="card">
                    <h3>Project READMEs</h3>
                    <p>README files for the project and its components.</p>
                    <p><a href="readme/index.html">View README Files</a></p>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">User Guides</h2>
            <div class="section-content">
                <div class="card">
                    <h3>Installation Guide</h3>
                    <p>Instructions for installing and setting up the Quantis platform.</p>
                    <p><a href="guides/installation_guide.html">View Installation Guide</a></p>
                </div>
                <div class="card">
                    <h3>User Guide</h3>
                    <p>Instructions for using the Quantis platform.</p>
                    <p><a href="guides/user_guide.html">View User Guide</a></p>
                </div>
                <div class="card">
                    <h3>API Guide</h3>
                    <p>Instructions for using the Quantis API.</p>
                    <p><a href="guides/api_guide.html">View API Guide</a></p>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>Generated on $(date) by documentation_generator.sh</p>
        </div>
    </div>
</body>
</html>
EOF

    echo -e "${GREEN}Index file generation completed.${NC}"
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    show_help
fi

while [ "$1" != "" ]; do
    case $1 in
        --all )     GEN_API=true
                    GEN_CODE=true
                    GEN_README=true
                    GEN_GUIDES=true
                    GEN_ALL=true
                    ;;
        --api )     GEN_API=true
                    ;;
        --code )    GEN_CODE=true
                    ;;
        --readme )  GEN_README=true
                    ;;
        --guides )  GEN_GUIDES=true
                    ;;
        --output )  shift
                    OUTPUT_DIR="$1"
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
echo -e "${BLUE}Starting Quantis documentation generation...${NC}"

# Check dependencies
check_dependencies

# Prepare output directory
prepare_output_dir

# Generate documentation
if $GEN_API || $GEN_ALL; then
    generate_api_docs
fi

if $GEN_CODE || $GEN_ALL; then
    generate_code_docs
fi

if $GEN_README || $GEN_ALL; then
    generate_readme_files
fi

if $GEN_GUIDES || $GEN_ALL; then
    generate_user_guides
fi

# Generate index file
generate_index

echo -e "${GREEN}Quantis documentation generation completed successfully!${NC}"
echo -e "Documentation is available at: $OUTPUT_DIR/index.html"
exit 0
