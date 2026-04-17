# Quantis

![CI/CD Status](https://img.shields.io/github/actions/workflow/status/quantsingularity/Quantis/cicd.yml?branch=main&label=CI/CD&logo=github)
[![Test Coverage](https://img.shields.io/badge/coverage-82%25-brightgreen)](https://github.com/quantsingularity/Quantis/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Quantitative Trading & Investment Analytics Platform

Quantis is a comprehensive quantitative trading and investment analytics platform that combines advanced statistical models, machine learning algorithms, and real-time market data to provide powerful insights and automated trading strategies.

<div align="center">
  <img src="docs/images/Quantis_dashboard.bmp" alt="Quantis Dashboard" width="80%">
</div>

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Quantis provides a robust platform for quantitative analysis, algorithmic trading, and investment portfolio optimization. The system leverages advanced statistical models and machine learning algorithms to analyze market data, identify trading opportunities, and execute automated trading strategies. The microservices architecture ensures that each component from data ingestion to strategy execution is independent, resilient, and can be scaled horizontally to handle high-volume, low-latency market operations.

---

## Project Structure

The project is organized into several main components:

```
Quantis/
├── code/                   # Core backend logic, services, and shared utilities
├── docs/                   # Project documentation
├── infrastructure/         # DevOps, deployment, and infra-related code
├── mobile-frontend/        # Mobile application
├── web-frontend/           # Web dashboard
├── scripts/                # Automation, setup, and utility scripts
├── LICENSE                 # License information
├── README.md               # Project overview and instructions
└── tools/                  # Formatter configs, linting tools, and dev utilities
```

## Key Features

Quantis's functionality is divided into four main domains, each powered by dedicated services.

### 1. Data Processing

- Real-time and historical market data
- Alternative data support
- Automated data validation

### 2. Quantitative Analysis

- Statistical and machine learning models
- Risk metrics including VaR
- Portfolio optimization tools

### 3. Trading Strategies

- Strategy creation and testing
- Signal generation
- Backtesting and execution

### 4. Portfolio Management

- Asset allocation
- Risk monitoring
- Performance tracking and rebalancing

---

## Architecture

Quantis follows a microservices architecture, logically grouped into three main service layers, supported by a common infrastructure.

### Architectural Components

| Layer                   | Key Services                                                                  | Description                                                                           |
| :---------------------- | :---------------------------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| **Data Services**       | Market Data, Alternative Data, Historical Data, Data Quality                  | Responsible for all data ingestion, storage, cleaning, and retrieval.                 |
| **Analytical Services** | Statistical Analysis, Machine Learning, Risk Analysis, Portfolio Optimization | Contains the core quantitative intelligence, running models and generating insights.  |
| **Trading Services**    | Strategy, Signal Generation, Backtesting, Execution                           | Manages the full lifecycle of trading strategies, from development to live execution. |
| **Infrastructure**      | API Gateway, Authentication Service, Monitoring Stack, Data Storage           | Provides common technical capabilities and ensures system stability and security.     |

---

## Technology Stack

The platform is built using a modern, performant, and well-supported technology stack.

| Category               | Key Technologies                                                   | Description                                                                                                                        |
| :--------------------- | :----------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| **Backend**            | Python, Rust, FastAPI, Flask                                       | Python for data science and rapid development; Rust for performance-critical components. FastAPI/Flask for robust API development. |
| **Databases**          | PostgreSQL, InfluxDB                                               | PostgreSQL for relational data and core system state; InfluxDB for high-volume time series data storage.                           |
| **Task Queue**         | Celery, Redis                                                      | Celery for asynchronous task processing; Redis for task queuing and caching.                                                       |
| **ML/Quant Libraries** | scikit-learn, PyTorch, pandas-ta, pyfolio, zipline                 | Specialized libraries for machine learning, technical analysis, performance reporting, and event-driven backtesting.               |
| **Frontend**           | React, TypeScript, Redux Toolkit, D3.js, Plotly, TradingView       | Modern stack for a responsive, data-rich web dashboard with advanced visualization capabilities.                                   |
| **DevOps**             | Docker, Kubernetes, GitHub Actions, Prometheus, Grafana, ELK Stack | Full-stack CI/CD, container orchestration, and observability tools for production readiness.                                       |

---

## Getting Started

### Prerequisites

To set up the platform, ensure you have the following installed:

- **Python** (v3.9+)
- **Node.js** (v16+)
- **Docker** and Docker Compose
- **Kubernetes** (for production deployment)

### Setup

The recommended way to set up the development environment is using the provided scripts:

| Step                    | Command                                                                   | Description                                                     |
| :---------------------- | :------------------------------------------------------------------------ | :-------------------------------------------------------------- |
| **1. Clone Repository** | `git clone https://github.com/quantsingularity/Quantis.git && cd Quantis` | Download the source code and navigate to the project directory. |
| **2. Run Setup Script** | `./setup_quantis_env.sh`                                                  | Installs dependencies and configures the local environment.     |
| **3. Start Services**   | `./run_quantis.sh dev`                                                    | Starts all core services for development.                       |

**Access Points:**

- **Web Dashboard**: `http://localhost:3000`
- **API Documentation**: `http://localhost:8000/docs`
- **Monitoring Dashboard**: `http://localhost:9090`

### Deployment

Quantis supports both Docker Compose for local environments and Kubernetes for production deployment.

| Deployment Target  | Command Example                        |
| :----------------- | :------------------------------------- |
| **Docker Compose** | `docker-compose up -d`                 |
| **Kubernetes**     | `kubectl apply -f infrastructure/k8s/` |

---

## API Documentation

Quantis exposes a comprehensive, versioned API for all platform interactions, accessible via the API Gateway.

### Key API Endpoints

| Service         | Endpoint                              | Method | Description                                 |
| :-------------- | :------------------------------------ | :----- | :------------------------------------------ |
| **Market Data** | `/api/v1/market/prices`               | `GET`  | Get real-time market prices.                |
| **Strategy**    | `/api/v1/strategies`                  | `POST` | Create a new trading strategy.              |
| **Strategy**    | `/api/v1/strategies/{id}/backtest`    | `POST` | Run a strategy backtest and return results. |
| **Portfolio**   | `/api/v1/portfolios/{id}/performance` | `GET`  | Get detailed portfolio performance metrics. |
| **Portfolio**   | `/api/v1/portfolios/{id}/rebalance`   | `POST` | Trigger an automated portfolio rebalancing. |

Full API documentation, including request/response schemas, is available at `http://localhost:8000/docs`.

---

## Testing

The project maintains an overall test coverage of **82%** across all components, ensuring reliability and accuracy in financial calculations and trading logic.

### Test Coverage Summary

| Component                | Coverage | Status |
| :----------------------- | :------- | :----- |
| **Trading Services**     | 87%      | ✅     |
| **Data Services**        | 85%      | ✅     |
| **Analytical Services**  | 83%      | ✅     |
| **Portfolio Management** | 80%      | ✅     |
| **API Layer**            | 90%      | ✅     |
| **Frontend Components**  | 75%      | ✅     |

### Testing Types

Testing is categorized into four main types:

| Test Type         | Description                                                        |
| ----------------- | ------------------------------------------------------------------ |
| Unit Tests        | Validate individual functions, models, and core trading logic      |
| Integration Tests | Validate end-to-end workflows, APIs, and database interactions     |
| Performance Tests | Measure throughput, backtesting speed, and API response under load |
| Security Tests    | Ensure compliance with security standards and best practices       |

**Running Tests:** All tests can be executed using the `pytest` command from the root directory: `pytest`. Specific categories can be targeted (e.g., `pytest tests/unit/`).

---

## CI/CD Pipeline

Quantis uses GitHub Actions for continuous integration and deployment:

| Stage                | Control Area                    | Institutional-Grade Detail                                                              |
| :------------------- | :------------------------------ | :-------------------------------------------------------------------------------------- |
| **Formatting Check** | Change Triggers                 | Enforced on all `push` and `pull_request` events to `main` and `develop`                |
|                      | Manual Oversight                | On-demand execution via controlled `workflow_dispatch`                                  |
|                      | Source Integrity                | Full repository checkout with complete Git history for auditability                     |
|                      | Python Runtime Standardization  | Python 3.10 with deterministic dependency caching                                       |
|                      | Backend Code Hygiene            | `autoflake` to detect unused imports/variables using non-mutating diff-based validation |
|                      | Backend Style Compliance        | `black --check` to enforce institutional formatting standards                           |
|                      | Non-Intrusive Validation        | Temporary workspace comparison to prevent unauthorized source modification              |
|                      | Node.js Runtime Control         | Node.js 18 with locked dependency installation via `npm ci`                             |
|                      | Web Frontend Formatting Control | Prettier checks for web-facing assets                                                   |
|                      | Mobile Frontend Formatting      | Prettier enforcement for mobile application codebases                                   |
|                      | Documentation Governance        | Repository-wide Markdown formatting enforcement                                         |
|                      | Infrastructure Configuration    | Prettier validation for YAML/YML infrastructure definitions                             |
|                      | Compliance Gate                 | Any formatting deviation fails the pipeline and blocks merge                            |

---

## Documentation

| Document                    | Path                 | Description                                                    |
| :-------------------------- | :------------------- | :------------------------------------------------------------- |
| **README**                  | `README.md`          | High-level overview, project scope, and repository entry point |
| **Installation Guide**      | `INSTALLATION.md`    | Step-by-step installation and environment setup                |
| **API Reference**           | `API.md`             | Detailed documentation for all API endpoints                   |
| **CLI Reference**           | `CLI.md`             | Command-line interface usage, commands, and examples           |
| **User Guide**              | `USAGE.md`           | Comprehensive end-user guide, workflows, and examples          |
| **Architecture Overview**   | `ARCHITECTURE.md`    | System architecture, components, and design rationale          |
| **Configuration Guide**     | `CONFIGURATION.md`   | Configuration options, environment variables, and tuning       |
| **Feature Matrix**          | `FEATURE_MATRIX.md`  | Feature coverage, capabilities, and roadmap alignment          |
| **Contributing Guidelines** | `CONTRIBUTING.md`    | Contribution workflow, coding standards, and PR requirements   |
| **Troubleshooting**         | `TROUBLESHOOTING.md` | Common issues, diagnostics, and remediation steps              |

## Contributing

We welcome contributions to Quantis! To get involved:

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/your-feature-name`).
3.  Commit your changes and push to the branch.
4.  Open a Pull Request for review.

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
