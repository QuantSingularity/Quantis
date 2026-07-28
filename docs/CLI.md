# CLI Reference

Command-line interface reference for Quantis platform scripts and utilities.

## Table of Contents

- [Overview](#overview)
- [Installation & Setup Commands](#installation--setup-commands)
- [Development Commands](#development-commands)
- [Testing Commands](#testing-commands)
- [Build & Deployment Commands](#build--deployment-commands)
- [Monitoring Commands](#monitoring-commands)
- [Utility Commands](#utility-commands)
- [Environment Variables](#environment-variables)

---

## Overview

Quantis provides several shell scripts for managing the platform. All scripts are located in the `scripts/` directory.

**Make scripts executable:**

```bash
chmod +x scripts/*.sh
# Or use the provided script:
./scripts/make_scripts_executable.sh
```

---

## Installation & Setup Commands

### setup_quantis_env.sh

**Purpose**: Automated environment setup for Quantis platform.

**Usage:**

```bash
./scripts/setup_quantis_env.sh
```

**What it does:**

- Creates Python virtual environments for API and models
- Installs all Python dependencies from requirements.txt
- Installs Node.js dependencies for frontend
- Configures initial environment variables
- Verifies installation integrity

**Options:**

| Flag   | Description          | Example                          |
| ------ | -------------------- | -------------------------------- |
| (none) | Full automated setup | `./scripts/setup_quantis_env.sh` |

**Expected Output:**

```
Starting Quantis project setup...
Changed directory to /Quantis
Setting up Quantis Backend API...
Creating Python virtual environment for API...
Installing API Python dependencies...
API dependencies installed.
Setting up Quantis Web Frontend...
Installing Web Frontend Node.js dependencies...
Setup complete!
```

---

### setup_environment.sh

**Purpose**: Configure environment variables and settings.

**Usage:**

```bash
./scripts/setup_environment.sh [OPTIONS]
```

**Arguments:**

| Command                | Arguments | Description                   | Example                                 |
| ---------------------- | --------- | ----------------------------- | --------------------------------------- |
| `setup_environment.sh` | -         | Interactive environment setup | `./scripts/setup_environment.sh`        |
| `setup_environment.sh` | `--prod`  | Production environment        | `./scripts/setup_environment.sh --prod` |
| `setup_environment.sh` | `--dev`   | Development environment       | `./scripts/setup_environment.sh --dev`  |

---

## Development Commands

### run_quantis.sh

**Purpose**: Start all Quantis services for development.

**Usage:**

```bash
./scripts/run_quantis.sh [MODE]
```

**Arguments:**

| Command          | Arguments | Description                       | Example                         |
| ---------------- | --------- | --------------------------------- | ------------------------------- |
| `run_quantis.sh` | -         | Start all services (default mode) | `./scripts/run_quantis.sh`      |
| `run_quantis.sh` | `dev`     | Development mode with auto-reload | `./scripts/run_quantis.sh dev`  |
| `run_quantis.sh` | `prod`    | Production mode                   | `./scripts/run_quantis.sh prod` |

**Services Started:**

- API server (FastAPI) on port 8000
- Model service (ML inference) on port 8001
- Frontend (React) on port 3000
- Redis (caching) on port 6379
- Celery workers for async tasks

**Example Output:**

```bash
Starting Quantis application...
Creating Python virtual environment...
Starting API server...
INFO:     Uvicorn running on http://0.0.0.0:8000
Starting model service...
Model service running with PID: 12345
Starting frontend...
Compiled successfully!
Frontend running with PID: 12346
Quantis application is running!
Access the application at: http://localhost:3000
Press Ctrl+C to stop all services
```

**Stopping Services:**
Press `Ctrl+C` to gracefully stop all services.

---

## Testing Commands

### test_quantis.sh

**Purpose**: Run comprehensive test suite.

**Usage:**

```bash
./scripts/test_quantis.sh [OPTIONS]
```

**Arguments:**

| Command           | Arguments       | Description                | Example                                   |
| ----------------- | --------------- | -------------------------- | ----------------------------------------- |
| `test_quantis.sh` | -               | Run all tests              | `./scripts/test_quantis.sh`               |
| `test_quantis.sh` | `--unit`        | Run unit tests only        | `./scripts/test_quantis.sh --unit`        |
| `test_quantis.sh` | `--integration` | Run integration tests only | `./scripts/test_quantis.sh --integration` |
| `test_quantis.sh` | `--coverage`    | Run with coverage report   | `./scripts/test_quantis.sh --coverage`    |

**Example:**

```bash
# Run all tests
./scripts/test_quantis.sh

# Run unit tests with coverage
./scripts/test_quantis.sh --unit --coverage
```

---

### test_runner.sh

**Purpose**: Advanced test runner with filtering and reporting.

**Usage:**

```bash
./scripts/test_runner.sh [OPTIONS]
```

**Arguments:**

| Command          | Arguments    | Description                | Example                                  |
| ---------------- | ------------ | -------------------------- | ---------------------------------------- |
| `test_runner.sh` | `-k PATTERN` | Run tests matching pattern | `./scripts/test_runner.sh -k "test_api"` |
| `test_runner.sh` | `-m MARKER`  | Run tests with marker      | `./scripts/test_runner.sh -m "slow"`     |
| `test_runner.sh` | `--verbose`  | Verbose output             | `./scripts/test_runner.sh --verbose`     |
| `test_runner.sh` | `--failfast` | Stop on first failure      | `./scripts/test_runner.sh --failfast`    |

**Test Markers:**

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.api` - API tests
- `@pytest.mark.ml` - ML model tests

---

## Build & Deployment Commands

### build_quantis.sh

**Purpose**: Build production artifacts.

**Usage:**

```bash
./scripts/build_quantis.sh [OPTIONS]
```

**Arguments:**

| Command            | Arguments    | Description          | Example                                 |
| ------------------ | ------------ | -------------------- | --------------------------------------- |
| `build_quantis.sh` | -            | Build all components | `./scripts/build_quantis.sh`            |
| `build_quantis.sh` | `--frontend` | Build frontend only  | `./scripts/build_quantis.sh --frontend` |
| `build_quantis.sh` | `--backend`  | Build backend only   | `./scripts/build_quantis.sh --backend`  |
| `build_quantis.sh` | `--docker`   | Build Docker images  | `./scripts/build_quantis.sh --docker`   |

**Example:**

```bash
# Build everything
./scripts/build_quantis.sh

# Build and push Docker images
./scripts/build_quantis.sh --docker --push
```

---

### unified_build.sh

**Purpose**: Unified build process for CI/CD.

**Usage:**

```bash
./scripts/unified_build.sh [ENVIRONMENT]
```

**Arguments:**

| Command            | Arguments | Description       | Example                              |
| ------------------ | --------- | ----------------- | ------------------------------------ |
| `unified_build.sh` | `dev`     | Development build | `./scripts/unified_build.sh dev`     |
| `unified_build.sh` | `staging` | Staging build     | `./scripts/unified_build.sh staging` |
| `unified_build.sh` | `prod`    | Production build  | `./scripts/unified_build.sh prod`    |

**Build Steps:**

1. Install dependencies
2. Run linters and formatters
3. Run test suite
4. Build frontend assets
5. Create Docker images
6. Tag and push to registry (if configured)

---

## Monitoring Commands

### monitoring_dashboard.sh

**Purpose**: Start monitoring dashboard (Prometheus + Grafana).

**Usage:**

```bash
./scripts/monitoring_dashboard.sh [ACTION]
```

**Arguments:**

| Command                   | Arguments | Description            | Example                                     |
| ------------------------- | --------- | ---------------------- | ------------------------------------------- |
| `monitoring_dashboard.sh` | `start`   | Start monitoring stack | `./scripts/monitoring_dashboard.sh start`   |
| `monitoring_dashboard.sh` | `stop`    | Stop monitoring stack  | `./scripts/monitoring_dashboard.sh stop`    |
| `monitoring_dashboard.sh` | `restart` | Restart monitoring     | `./scripts/monitoring_dashboard.sh restart` |
| `monitoring_dashboard.sh` | `logs`    | View logs              | `./scripts/monitoring_dashboard.sh logs`    |

**Access Points:**

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default login: admin/admin)

**Example:**

```bash
# Start monitoring
./scripts/monitoring_dashboard.sh start

# View logs
./scripts/monitoring_dashboard.sh logs -f
```

---

## Utility Commands

### linting.sh

**Purpose**: Run code quality checks and formatters.

**Usage:**

```bash
./scripts/linting.sh [OPTIONS]
```

**Arguments:**

| Command      | Arguments      | Description                         | Example                             |
| ------------ | -------------- | ----------------------------------- | ----------------------------------- |
| `linting.sh` | -              | Run all linters                     | `./scripts/linting.sh`              |
| `linting.sh` | `--fix`        | Auto-fix issues                     | `./scripts/linting.sh --fix`        |
| `linting.sh` | `--python`     | Python only (pylint, flake8, black) | `./scripts/linting.sh --python`     |
| `linting.sh` | `--javascript` | JavaScript only (ESLint, Prettier)  | `./scripts/linting.sh --javascript` |

**Tools Used:**

- **Python**: black, isort, flake8, pylint, mypy
- **JavaScript**: ESLint, Prettier
- **Markdown**: markdownlint

---

### lint-all.sh

**Purpose**: Comprehensive linting across all file types.

**Usage:**

```bash
./scripts/lint-all.sh
```

**Checks:**

- Python code style and type hints
- JavaScript/TypeScript code style
- Markdown formatting
- YAML syntax
- Shell script quality (shellcheck)

---

### data_processor.sh

**Purpose**: Data processing and ETL operations.

**Usage:**

```bash
./scripts/data_processor.sh [ACTION] [OPTIONS]
```

**Arguments:**

| Command             | Arguments        | Description           | Example                                          |
| ------------------- | ---------------- | --------------------- | ------------------------------------------------ |
| `data_processor.sh` | `process FILE`   | Process data file     | `./scripts/data_processor.sh process data.csv`   |
| `data_processor.sh` | `validate FILE`  | Validate data format  | `./scripts/data_processor.sh validate data.csv`  |
| `data_processor.sh` | `clean FILE`     | Clean and normalize   | `./scripts/data_processor.sh clean data.csv`     |
| `data_processor.sh` | `transform FILE` | Apply transformations | `./scripts/data_processor.sh transform data.csv` |

**Example:**

```bash
# Process and clean stock data
./scripts/data_processor.sh process stock_prices.csv --clean --normalize

# Validate dataset format
./scripts/data_processor.sh validate dataset.parquet
```

---

### documentation_generator.sh

**Purpose**: Generate API documentation and code references.

**Usage:**

```bash
./scripts/documentation_generator.sh [OPTIONS]
```

**Arguments:**

| Command                      | Arguments | Description       | Example                                       |
| ---------------------------- | --------- | ----------------- | --------------------------------------------- |
| `documentation_generator.sh` | -         | Generate all docs | `./scripts/documentation_generator.sh`        |
| `documentation_generator.sh` | `--api`   | API docs only     | `./scripts/documentation_generator.sh --api`  |
| `documentation_generator.sh` | `--code`  | Code reference    | `./scripts/documentation_generator.sh --code` |

**Generated Files:**

- API documentation (OpenAPI/Swagger)
- Code reference documentation
- Module dependency graphs
- Architecture diagrams

---

## Environment Variables

Key environment variables used by CLI commands:

| Variable                | Description         | Default                    | Example            |
| ----------------------- | ------------------- | -------------------------- | ------------------ |
| `QUANTIS_ENV`           | Environment mode    | `development`              | `production`       |
| `QUANTIS_API_PORT`      | API server port     | `8000`                     | `8080`             |
| `QUANTIS_FRONTEND_PORT` | Frontend port       | `3000`                     | `3001`             |
| `DATABASE_URL`          | Database connection | `sqlite:///./quantis.db`   | `postgresql://...` |
| `REDIS_URL`             | Redis connection    | `redis://localhost:6379/0` | `redis://...`      |
| `LOG_LEVEL`             | Logging level       | `INFO`                     | `DEBUG`            |
| `PYTHONPATH`            | Python module path  | -                          | `/app/code`        |

**Setting Environment Variables:**

```bash
# Temporary (current session)
export QUANTIS_ENV=production
export LOG_LEVEL=DEBUG

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export QUANTIS_ENV=production' >> ~/.bashrc

# Using .env file (recommended)
cat > .env << EOF
QUANTIS_ENV=production
LOG_LEVEL=INFO
DATABASE_URL=postgresql://user:pass@localhost/quantis
EOF
```

---

## Docker Commands

While not shell scripts, Docker commands are part of the CLI workflow:

### Build Docker Images

```bash
# Build all images
docker-compose build

# Build specific service
docker-compose build api
```

### Start Services

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d api
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
```

### Execute Commands in Container

```bash
# Run bash in API container
docker-compose exec api bash

# Run pytest in API container
docker-compose exec api pytest tests/
```

---

## Kubernetes Commands

For production Kubernetes deployments:

### Deploy Application

```bash
# Apply all manifests
kubectl apply -k infrastructure/kubernetes/

# Apply specific environment
kubectl apply -k infrastructure/kubernetes/environments/prod
```

### Check Status

```bash
# Get pods
kubectl get pods -n quantis

# Get services
kubectl get services -n quantis

# Describe pod
kubectl describe pod <pod-name> -n quantis
```

### View Logs

```bash
# Stream logs from pod
kubectl logs -f <pod-name> -n quantis

# View logs from all pods with label
kubectl logs -f -l app=quantis-api -n quantis
```

### Port Forwarding

```bash
# Forward API port
kubectl port-forward -n quantis svc/api-service 8000:8000

# Forward frontend port
kubectl port-forward -n quantis svc/frontend-service 3000:80
```

---

## Common Workflows

### Development Workflow

```bash
# 1. Initial setup
./scripts/setup_quantis_env.sh

# 2. Start development servers
./scripts/run_quantis.sh dev

# 3. Run tests (in another terminal)
./scripts/test_quantis.sh

# 4. Check code quality
./scripts/linting.sh --fix

# 5. Process data (if needed)
./scripts/data_processor.sh process data.csv
```

### CI/CD Workflow

```bash
# 1. Lint code
./scripts/lint-all.sh

# 2. Run tests with coverage
./scripts/test_quantis.sh --coverage

# 3. Build artifacts
./scripts/unified_build.sh prod

# 4. Deploy (automated in CI/CD)
kubectl apply -k infrastructure/kubernetes/environments/prod
```

### Monitoring Workflow

```bash
# 1. Start monitoring stack
./scripts/monitoring_dashboard.sh start

# 2. Start application
./scripts/run_quantis.sh prod

# 3. View metrics
# Navigate to http://localhost:9090 (Prometheus)
# Navigate to http://localhost:3000 (Grafana)

# 4. View logs
docker-compose logs -f api
```

---

## Troubleshooting CLI Issues

### Permission Denied

**Issue**: `Permission denied` when running scripts

**Solution**:

```bash
chmod +x scripts/*.sh
# Or
./scripts/make_scripts_executable.sh
```

### Command Not Found

**Issue**: `command not found` errors

**Solution**:

```bash
# Ensure you're in the project root
cd /path/to/Quantis

# Run with explicit path
./scripts/script_name.sh

# Or add scripts to PATH
export PATH="$PATH:$(pwd)/scripts"
```

### Python Module Errors

**Issue**: `ModuleNotFoundError` when running scripts

**Solution**:

```bash
# Activate virtual environment
source code/api/venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH=/path/to/Quantis/code:$PYTHONPATH
```

---

## Quick Reference

| Task                     | Command                                    |
| ------------------------ | ------------------------------------------ |
| **Initial setup**        | `./scripts/setup_quantis_env.sh`           |
| **Start development**    | `./scripts/run_quantis.sh dev`             |
| **Run tests**            | `./scripts/test_quantis.sh`                |
| **Lint code**            | `./scripts/linting.sh --fix`               |
| **Build for production** | `./scripts/build_quantis.sh`               |
| **Start monitoring**     | `./scripts/monitoring_dashboard.sh start`  |
| **Process data**         | `./scripts/data_processor.sh process FILE` |
| **Generate docs**        | `./scripts/documentation_generator.sh`     |

---
