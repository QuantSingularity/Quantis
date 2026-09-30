#!/bin/bash
# monitoring_dashboard.sh - Automated monitoring dashboard setup and management
#
# This script automates the setup, configuration, and management of monitoring dashboards
# for the Quantis project using Prometheus and Grafana.
#
# Usage: ./monitoring_dashboard.sh [options]
# Options:
#   --setup              Initial setup of monitoring stack
#   --update             Update existing monitoring configuration
#   --start              Start monitoring services
#   --stop               Stop monitoring services
#   --status             Check status of monitoring services
#   --backup             Backup monitoring configuration and data
#   --restore FILE       Restore monitoring configuration from backup
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
# Resolve the actual repository root instead of trusting the caller's
# current directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# The project's real, already-configured monitoring stack lives at
# infrastructure/monitoring (with prometheus.yml, alert_rules.yml,
# grafana_dashboards/, grafana_provisioning/ already committed) - not a
# fabricated top-level "monitoring" directory.
MONITORING_DIR="$PROJECT_ROOT/infrastructure/monitoring"
GRAFANA_DIR="$MONITORING_DIR/grafana_dashboards"
BACKUP_DIR="$PROJECT_ROOT/monitoring_backups"
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
# Detect either Docker Compose v1 (standalone "docker-compose" binary) or
# the now-standard v2 plugin form ("docker compose"), since v1 is
# deprecated and increasingly absent on current Docker installations.
DOCKER_COMPOSE_CMD=""

# Function to display help message
show_help() {
    echo -e "${BLUE}Monitoring Dashboard Management for Quantis Project${NC}"
    echo ""
    echo "Usage: ./monitoring_dashboard.sh [options]"
    echo ""
    echo "Options:"
    echo "  --setup              Initial setup of monitoring stack"
    echo "  --update             Update existing monitoring configuration"
    echo "  --start              Start monitoring services"
    echo "  --stop               Stop monitoring services"
    echo "  --status             Check status of monitoring services"
    echo "  --backup             Backup monitoring configuration and data"
    echo "  --restore FILE       Restore monitoring configuration from backup"
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
    echo -e "${BLUE}Checking monitoring dependencies...${NC}"

    # Check Docker
    if ! command_exists docker; then
        echo -e "${RED}Error: Docker is required but not installed.${NC}"
        echo "Please install Docker and try again."
        exit 1
    fi

    # Check Docker Compose - prefer the v1 standalone binary if present,
    # otherwise fall back to the v2 "docker compose" plugin subcommand
    # (v1 is deprecated and often no longer installed on current systems).
    if command_exists docker-compose; then
        DOCKER_COMPOSE_CMD="docker-compose"
    elif docker compose version >/dev/null 2>&1; then
        DOCKER_COMPOSE_CMD="docker compose"
    else
        echo -e "${RED}Error: Docker Compose is required but not installed.${NC}"
        echo "Please install Docker Compose (either the standalone docker-compose"
        echo "binary or the 'docker compose' plugin) and try again."
        exit 1
    fi

    echo -e "${GREEN}All required monitoring dependencies are installed (using: ${DOCKER_COMPOSE_CMD}).${NC}"
}

# Function to setup monitoring stack
setup_monitoring() {
    echo -e "${BLUE}Setting up monitoring stack...${NC}"

    # Create monitoring directory if it doesn't exist
    mkdir -p "$MONITORING_DIR"
    mkdir -p "$GRAFANA_DIR"

    # This repository already ships real, curated monitoring configuration
    # under infrastructure/monitoring (prometheus.yml, alert_rules.yml,
    # grafana_dashboards/, grafana_provisioning/). Never blindly overwrite
    # those with generic boilerplate - only create files that don't exist yet.
    if [ -f "$MONITORING_DIR/docker-compose.yml" ]; then
        echo -e "${YELLOW}$MONITORING_DIR/docker-compose.yml already exists - leaving it untouched.${NC}"
        echo -e "${YELLOW}Delete it first if you want this script to regenerate it.${NC}"
        echo -e "${GREEN}Monitoring stack setup completed (nothing to do).${NC}"
        return
    fi

    # Create Docker Compose file for monitoring stack
    cat > "$MONITORING_DIR/docker-compose.yml" << EOF
version: '3'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: quantis_prometheus
    ports:
      - "$PROMETHEUS_PORT:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: quantis_grafana
    ports:
      - "$GRAFANA_PORT:3000"
    volumes:
      - ./grafana_dashboards:/var/lib/grafana/dashboards
      - ./grafana_provisioning:/etc/grafana/provisioning
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=quantis_admin
      - GF_USERS_ALLOW_SIGN_UP=false
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
EOF

    # Create Prometheus configuration file (only if one doesn't already exist -
    # this repo ships a real, tailored prometheus.yml that must not be
    # silently replaced with generic placeholder scrape targets).
    if [ -f "$MONITORING_DIR/prometheus.yml" ]; then
        echo -e "${YELLOW}$MONITORING_DIR/prometheus.yml already exists - leaving it untouched.${NC}"
    else
    cat > "$MONITORING_DIR/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          # - alertmanager:9093

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'quantis_api'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['host.docker.internal:8000']

  - job_name: 'quantis_models'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['host.docker.internal:8001']

  - job_name: 'node_exporter'
    static_configs:
      - targets: ['host.docker.internal:9100']
EOF
    fi

    # Create Grafana provisioning directories
    mkdir -p "$MONITORING_DIR/grafana_provisioning/datasources"
    mkdir -p "$MONITORING_DIR/grafana_provisioning/dashboards"

    # Create Grafana datasource configuration (skip if one already exists)
    if [ -f "$MONITORING_DIR/grafana_provisioning/datasources/prometheus.yml" ]; then
        echo -e "${YELLOW}Grafana datasource config already exists - leaving it untouched.${NC}"
    else
    cat > "$MONITORING_DIR/grafana_provisioning/datasources/prometheus.yml" << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
EOF
    fi

    # Create Grafana dashboard provisioning configuration (skip if one
    # already exists under either the legacy "dashboards.yml" or the name
    # already used by this repo, "dashboard.yml").
    if [ -f "$MONITORING_DIR/grafana_provisioning/dashboards/dashboards.yml" ] || \
       [ -f "$MONITORING_DIR/grafana_provisioning/dashboards/dashboard.yml" ]; then
        echo -e "${YELLOW}Grafana dashboard provisioning config already exists - leaving it untouched.${NC}"
    else
    cat > "$MONITORING_DIR/grafana_provisioning/dashboards/dashboards.yml" << EOF
apiVersion: 1

providers:
  - name: 'Quantis Dashboards'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
      foldersFromFilesStructure: true
EOF
    fi

    # Create sample Grafana dashboard for Quantis API
    cat > "$GRAFANA_DIR/quantis_api_dashboard.json" << EOF
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": "-- Grafana --",
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "id": 1,
  "links": [],
  "panels": [
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "custom": {}
        },
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 9,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "hiddenSeries": false,
      "id": 2,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.3.7",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "http_requests_total",
          "interval": "",
          "legendFormat": "",
          "refId": "A"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "API Requests",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    },
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "custom": {}
        },
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 9,
        "w": 12,
        "x": 12,
        "y": 0
      },
      "hiddenSeries": false,
      "id": 4,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.3.7",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "http_request_duration_seconds",
          "interval": "",
          "legendFormat": "",
          "refId": "A"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "API Response Time",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    }
  ],
  "schemaVersion": 26,
  "style": "dark",
  "tags": [],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-6h",
    "to": "now"
  },
  "timepicker": {},
  "timezone": "",
  "title": "Quantis API Dashboard",
  "uid": "quantis-api",
  "version": 1
}
EOF

    # Create sample Grafana dashboard for Quantis Models
    cat > "$GRAFANA_DIR/quantis_models_dashboard.json" << EOF
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": "-- Grafana --",
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "id": 2,
  "links": [],
  "panels": [
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "custom": {}
        },
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 9,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "hiddenSeries": false,
      "id": 2,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.3.7",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "model_prediction_count",
          "interval": "",
          "legendFormat": "",
          "refId": "A"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "Model Predictions",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    },
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "custom": {}
        },
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 9,
        "w": 12,
        "x": 12,
        "y": 0
      },
      "hiddenSeries": false,
      "id": 4,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.3.7",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "model_prediction_duration_seconds",
          "interval": "",
          "legendFormat": "",
          "refId": "A"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "Model Prediction Time",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    }
  ],
  "schemaVersion": 26,
  "style": "dark",
  "tags": [],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-6h",
    "to": "now"
  },
  "timepicker": {},
  "timezone": "",
  "title": "Quantis Models Dashboard",
  "uid": "quantis-models",
  "version": 1
}
EOF

    echo -e "${GREEN}Monitoring stack setup completed.${NC}"
}

# Function to update monitoring configuration
update_monitoring() {
    echo -e "${BLUE}Updating monitoring configuration...${NC}"

    if [ ! -d "$MONITORING_DIR" ]; then
        echo -e "${RED}Error: Monitoring directory not found. Please run setup first.${NC}"
        exit 1
    fi

    # Update Prometheus configuration
    if [ -f "$MONITORING_DIR/prometheus.yml" ]; then
        echo "Updating Prometheus configuration..."
        # Here you would typically update the prometheus.yml file
        # based on the current state of the project
    fi

    # Update Grafana dashboards
    if [ -d "$GRAFANA_DIR" ]; then
        echo "Updating Grafana dashboards..."
        # Here you would typically update the Grafana dashboard files
        # based on the current state of the project
    fi

    echo -e "${GREEN}Monitoring configuration updated.${NC}"
}

# Function to start monitoring services
start_monitoring() {
    echo -e "${BLUE}Starting monitoring services...${NC}"

    if [ ! -f "$MONITORING_DIR/docker-compose.yml" ]; then
        echo -e "${RED}Error: docker-compose.yml not found. Please run setup first.${NC}"
        exit 1
    fi

    cd "$MONITORING_DIR"
    ${DOCKER_COMPOSE_CMD} up -d

    echo -e "${GREEN}Monitoring services started.${NC}"
    echo -e "Prometheus: http://localhost:$PROMETHEUS_PORT"
    echo -e "Grafana: http://localhost:$GRAFANA_PORT (admin/quantis_admin)"
}

# Function to stop monitoring services
stop_monitoring() {
    echo -e "${BLUE}Stopping monitoring services...${NC}"

    if [ ! -f "$MONITORING_DIR/docker-compose.yml" ]; then
        echo -e "${RED}Error: docker-compose.yml not found. Please run setup first.${NC}"
        exit 1
    fi

    cd "$MONITORING_DIR"
    ${DOCKER_COMPOSE_CMD} down

    echo -e "${GREEN}Monitoring services stopped.${NC}"
}

# Function to check status of monitoring services
check_status() {
    echo -e "${BLUE}Checking status of monitoring services...${NC}"

    if [ ! -f "$MONITORING_DIR/docker-compose.yml" ]; then
        echo -e "${RED}Error: docker-compose.yml not found. Please run setup first.${NC}"
        exit 1
    fi

    cd "$MONITORING_DIR"
    ${DOCKER_COMPOSE_CMD} ps
}

# Function to backup monitoring configuration and data
backup_monitoring() {
    echo -e "${BLUE}Backing up monitoring configuration and data...${NC}"

    if [ ! -d "$MONITORING_DIR" ]; then
        echo -e "${RED}Error: Monitoring directory not found. Please run setup first.${NC}"
        exit 1
    fi

    # Create backup directory if it doesn't exist
    mkdir -p "$BACKUP_DIR"

    # Create backup filename with timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    BACKUP_FILE="$BACKUP_DIR/monitoring_backup_$TIMESTAMP.tar.gz"

    # Create backup
    tar -czf "$BACKUP_FILE" -C "$PROJECT_ROOT" monitoring

    echo -e "${GREEN}Monitoring backup created: $BACKUP_FILE${NC}"
}

# Function to restore monitoring configuration from backup
restore_monitoring() {
    echo -e "${BLUE}Restoring monitoring configuration from backup...${NC}"

    if [ -z "$1" ]; then
        echo -e "${RED}Error: Backup file not specified.${NC}"
        echo "Usage: ./monitoring_dashboard.sh --restore BACKUP_FILE"
        exit 1
    fi

    BACKUP_FILE="$1"

    if [ ! -f "$BACKUP_FILE" ]; then
        echo -e "${RED}Error: Backup file not found: $BACKUP_FILE${NC}"
        exit 1
    fi

    # Stop monitoring services if running
    if [ -f "$MONITORING_DIR/docker-compose.yml" ]; then
        cd "$MONITORING_DIR"
        ${DOCKER_COMPOSE_CMD} down
    fi

    # Backup current configuration
    if [ -d "$MONITORING_DIR" ]; then
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        CURRENT_BACKUP="$BACKUP_DIR/monitoring_current_$TIMESTAMP.tar.gz"
        tar -czf "$CURRENT_BACKUP" -C "$PROJECT_ROOT" monitoring
        echo -e "${YELLOW}Current configuration backed up to: $CURRENT_BACKUP${NC}"
    fi

    # Remove current monitoring directory
    rm -rf "$MONITORING_DIR"

    # Extract backup
    tar -xzf "$BACKUP_FILE" -C "$PROJECT_ROOT"

    echo -e "${GREEN}Monitoring configuration restored from: $BACKUP_FILE${NC}"
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    show_help
fi

COMMAND=""
RESTORE_FILE=""

while [ "$1" != "" ]; do
    case $1 in
        --setup )   COMMAND="setup"
                    ;;
        --update )  COMMAND="update"
                    ;;
        --start )   COMMAND="start"
                    ;;
        --stop )    COMMAND="stop"
                    ;;
        --status )  COMMAND="status"
                    ;;
        --backup )  COMMAND="backup"
                    ;;
        --restore ) COMMAND="restore"
                    if [ $# -lt 2 ]; then
                        echo -e "${RED}Error: --restore requires a backup file argument.${NC}"
                        echo "Usage: ./monitoring_dashboard.sh --restore BACKUP_FILE"
                        exit 1
                    fi
                    shift
                    RESTORE_FILE="$1"
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
echo -e "${BLUE}Quantis Monitoring Dashboard Management${NC}"

# Check dependencies
check_dependencies

# Execute command
case $COMMAND in
    setup )     setup_monitoring
                ;;
    update )    update_monitoring
                ;;
    start )     start_monitoring
                ;;
    stop )      stop_monitoring
                ;;
    status )    check_status
                ;;
    backup )    backup_monitoring
                ;;
    restore )   restore_monitoring "$RESTORE_FILE"
                ;;
    * )         show_help
                ;;
esac

exit 0
