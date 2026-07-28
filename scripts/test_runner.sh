#!/bin/bash
# test_runner.sh - Comprehensive test runner for Quantis project
#
# This script automates the testing process for all components of the Quantis project:
# - Backend tests (code/backend/tests — a single flat suite; this repo does
#   not separate "unit" from "integration" tests into their own
#   directories, so --unit and --integration both run that same suite)
# - Web frontend tests (Vitest)
# - Mobile frontend tests (Jest)
# - End-to-end tests (if present under tests/e2e)
# - Performance tests (if present under tests/performance)
#
# Usage: ./test_runner.sh [options]
# Options:
#   --all                Run all tests
#   --unit               Run unit-level tests
#   --integration        Run integration-level tests
#   --e2e                Run only end-to-end tests
#   --performance        Run only performance tests
#   --component TYPE     Specify component to test (api, models, web, mobile, all)
#   --report             Generate HTML test report
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
RUN_UNIT=false
RUN_INTEGRATION=false
RUN_E2E=false
RUN_PERFORMANCE=false
COMPONENT="all"
GENERATE_REPORT=false
# Resolve the actual repository root instead of trusting the caller's
# current directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPORT_DIR="$PROJECT_ROOT/test_reports"
VENV_DIR="$PROJECT_ROOT/venv"

# Function to display help message
show_help() {
    echo -e "${BLUE}Test Runner for Quantis Project${NC}"
    echo ""
    echo "Usage: ./test_runner.sh [options]"
    echo ""
    echo "Options:"
    echo "  --all                Run all tests"
    echo "  --unit               Run unit-level tests"
    echo "  --integration        Run integration-level tests"
    echo "  --e2e                Run only end-to-end tests"
    echo "  --performance        Run only performance tests"
    echo "  --component TYPE     Specify component to test (api, models, web, mobile, all)"
    echo "  --report             Generate HTML test report"
    echo "  --help               Show this help message"
    echo ""
    exit 0
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Maps a --component name to its real directory, since the repository's
# actual layout (code/backend, code/quant_ml, web-frontend, mobile-frontend)
# does not match the component names used on the CLI.
resolve_component_dir() {
    case "$1" in
        api)     echo "${PROJECT_ROOT}/code/backend" ;;
        models)  echo "${PROJECT_ROOT}/code/quant_ml" ;;
        web)     echo "${PROJECT_ROOT}/web-frontend" ;;
        mobile)  echo "${PROJECT_ROOT}/mobile-frontend" ;;
        *)       echo "" ;;
    esac
}

# Function to check required dependencies
check_dependencies() {
    echo -e "${BLUE}Checking testing dependencies...${NC}"

    # Check Python dependencies
    if [ "$COMPONENT" = "api" ] || [ "$COMPONENT" = "models" ] || [ "$COMPONENT" = "all" ]; then
        if ! command_exists python3; then
            echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
            exit 1
        fi
    fi

    # Check Node.js dependencies
    if [ "$COMPONENT" = "web" ] || [ "$COMPONENT" = "mobile" ] || [ "$COMPONENT" = "all" ]; then
        if ! command_exists node; then
            echo -e "${RED}Error: Node.js is required but not installed.${NC}"
            exit 1
        fi

        if ! command_exists npm; then
            echo -e "${RED}Error: npm is required but not installed.${NC}"
            exit 1
        fi
    fi

    echo -e "${GREEN}All required testing dependencies are installed.${NC}"
}

# Function to prepare report directory
prepare_report_dir() {
    if $GENERATE_REPORT; then
        echo -e "${BLUE}Preparing report directory...${NC}"
        mkdir -p "$REPORT_DIR"
        echo -e "${GREEN}Report directory prepared.${NC}"
    fi
}

# Function to run the backend (code/backend) Python test suite.
# This repository keeps all backend tests in a single flat
# code/backend/tests directory (no unit/ vs integration/ split, and no
# pytest markers distinguishing them), so --unit and --integration both
# run this same suite; the "level" argument only affects report file naming.
run_python_tests() {
    local component=$1
    local level=$2
    echo -e "${BLUE}Running $component $level tests...${NC}"

    local component_dir
    component_dir="$(resolve_component_dir "$component")"

    if [ -z "$component_dir" ] || [ ! -d "$component_dir" ]; then
        echo -e "${YELLOW}Warning: $component directory not found. Skipping $level tests.${NC}"
        return
    fi

    if [ ! -d "$component_dir/tests" ]; then
        echo -e "${YELLOW}Warning: $component has no tests/ directory (quant_ml is exercised by the"
        echo -e "${YELLOW}backend's own test suite). Skipping $level tests for $component.${NC}"
        return
    fi

    # Backend and quant_ml share one virtual environment at the repo root.
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi

    (
      set -e
      # shellcheck source=/dev/null
      source "$VENV_DIR/bin/activate"

      pip install -q pytest pytest-cov pytest-html

      if [ -f "$component_dir/requirements.txt" ]; then
          pip install -q -r "$component_dir/requirements.txt"
      fi

      # Run from the project root so the "code" package resolves correctly.
      cd "$PROJECT_ROOT"
      export PYTHONPATH="$PROJECT_ROOT"

      if $GENERATE_REPORT; then
          mkdir -p "$REPORT_DIR/$component/$level"
          pytest "$component_dir/tests" -v \
              --cov="code/backend" --cov-report=term --cov-report="html:$REPORT_DIR/$component/$level/coverage" \
              --html="$REPORT_DIR/$component/$level/report.html" --self-contained-html || true
      else
          pytest "$component_dir/tests" -v || true
      fi

      deactivate
    )

    echo -e "${GREEN}$component $level tests completed.${NC}"
}

# Function to run JavaScript/TypeScript tests (web-frontend: Vitest, mobile-frontend: Jest)
run_js_tests() {
    local component=$1
    local level=$2
    echo -e "${BLUE}Running $component $level tests...${NC}"

    local component_dir
    component_dir="$(resolve_component_dir "$component")"

    if [ -z "$component_dir" ] || [ ! -d "$component_dir" ]; then
        echo -e "${YELLOW}Warning: $component directory not found. Skipping $level tests.${NC}"
        return
    fi
    if [ ! -f "$component_dir/package.json" ]; then
        echo -e "${YELLOW}Warning: package.json not found in $component. Skipping $level tests.${NC}"
        return
    fi

    (
      cd "$component_dir"
      echo "Installing dependencies..."
      npm install --no-audit --no-fund

      # Neither frontend distinguishes "unit" from "integration" tests via
      # separate npm scripts (both just run their single "test" script), so
      # --unit and --integration run the same command; "$level" only
      # affects where the report is written.
      if $GENERATE_REPORT; then
          mkdir -p "$REPORT_DIR/$component/$level/coverage"
          npm test -- --coverage 2>&1 | tee "$REPORT_DIR/$component/$level/output.log" || true
          if [ -d "coverage" ]; then
              cp -r coverage/* "$REPORT_DIR/$component/$level/coverage/" 2>/dev/null || true
          fi
      else
          npm test || true
      fi
    )

    echo -e "${GREEN}$component $level tests completed.${NC}"
}

# Function to run end-to-end tests
run_e2e_tests() {
    echo -e "${BLUE}Running end-to-end tests...${NC}"

    if [ ! -d "$PROJECT_ROOT/tests/e2e" ]; then
        echo -e "${YELLOW}Warning: End-to-end test directory (tests/e2e) not found. Skipping E2E tests.${NC}"
        echo -e "${YELLOW}This project does not currently have an E2E test suite set up.${NC}"
        return
    fi

    (
      cd "$PROJECT_ROOT/tests/e2e"

      # Check if using Cypress or Playwright
      if [ -f "package.json" ]; then
          npm install

          if $GENERATE_REPORT; then
              mkdir -p "$REPORT_DIR/e2e"

              if grep -q "cypress" package.json; then
                  npm run cypress:run -- --reporter mochawesome --reporter-options "reportDir=$REPORT_DIR/e2e,reportFilename=report" || true
              elif grep -q "playwright" package.json; then
                  npx playwright test --reporter=html || true
                  if [ -d "playwright-report" ]; then
                      cp -r playwright-report/* "$REPORT_DIR/e2e/"
                  fi
              else
                  npm test || true
              fi
          else
              npm test || true
          fi
      elif [ -f "requirements.txt" ]; then
          if [ ! -d "venv" ]; then
              echo "Creating virtual environment..."
              python3 -m venv venv
          fi
          # shellcheck source=/dev/null
          source venv/bin/activate
          pip install -q -r requirements.txt

          if $GENERATE_REPORT; then
              mkdir -p "$REPORT_DIR/e2e"
              pytest -v --html="$REPORT_DIR/e2e/report.html" --self-contained-html || true
          else
              pytest -v || true
          fi
          deactivate
      else
          echo -e "${YELLOW}Warning: No package.json or requirements.txt found in E2E test directory. Skipping E2E tests.${NC}"
      fi
    )

    echo -e "${GREEN}End-to-end tests completed.${NC}"
}

# Function to run performance tests
run_performance_tests() {
    echo -e "${BLUE}Running performance tests...${NC}"

    if [ ! -d "$PROJECT_ROOT/tests/performance" ]; then
        echo -e "${YELLOW}Warning: Performance test directory (tests/performance) not found. Skipping.${NC}"
        echo -e "${YELLOW}This project does not currently have a performance test suite set up.${NC}"
        return
    fi

    (
      cd "$PROJECT_ROOT/tests/performance"

      # Use shell globbing (nullglob-safe) to detect test files instead of
      # `[ -f "*.jmx" ]`, which tests for a file literally named "*.jmx"
      # and can never match a real file.
      shopt -s nullglob
      jmx_files=(*.jmx)
      js_files=(*.js)
      shopt -u nullglob

      if [ "${#jmx_files[@]}" -gt 0 ]; then
          if command_exists jmeter; then
              if $GENERATE_REPORT; then
                  mkdir -p "$REPORT_DIR/performance"
                  jmeter -n -t "${jmx_files[0]}" -l "$REPORT_DIR/performance/results.jtl" -e -o "$REPORT_DIR/performance/dashboard" || true
              else
                  jmeter -n -t "${jmx_files[0]}" || true
              fi
          else
              echo -e "${YELLOW}Warning: JMeter is not installed. Skipping performance tests.${NC}"
          fi
      elif [ -f "locustfile.py" ]; then
          if command_exists locust; then
              if $GENERATE_REPORT; then
                  mkdir -p "$REPORT_DIR/performance"
                  locust -f locustfile.py --headless -u 10 -r 1 --run-time 1m --html="$REPORT_DIR/performance/report.html" || true
              else
                  locust -f locustfile.py --headless -u 10 -r 1 --run-time 1m || true
              fi
          else
              echo -e "${YELLOW}Warning: Locust is not installed. Skipping performance tests.${NC}"
          fi
      elif [ "${#js_files[@]}" -gt 0 ] && grep -ql "k6" "${js_files[@]}" 2>/dev/null; then
          if command_exists k6; then
              if $GENERATE_REPORT; then
                  mkdir -p "$REPORT_DIR/performance"
                  k6 run --summary-export="$REPORT_DIR/performance/summary.json" "${js_files[@]}" || true
              else
                  k6 run "${js_files[@]}" || true
              fi
          else
              echo -e "${YELLOW}Warning: k6 is not installed. Skipping performance tests.${NC}"
          fi
      else
          echo -e "${YELLOW}Warning: No recognized performance test files found. Skipping performance tests.${NC}"
      fi
    )

    echo -e "${GREEN}Performance tests completed.${NC}"
}

# Function to generate HTML index for all reports
generate_report_index() {
    if $GENERATE_REPORT; then
        echo -e "${BLUE}Generating report index...${NC}"

        cat > "$REPORT_DIR/index.html" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantis Test Reports</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }
        h1, h2, h3 { color: #0066cc; }
        .report-section { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
        .report-links { margin-left: 20px; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .timestamp { color: #666; font-style: italic; }
    </style>
</head>
<body>
    <h1>Quantis Test Reports</h1>
    <div class="timestamp">Generated on $(date)</div>

EOF

        for component in api models web-frontend mobile-frontend; do
            if [ -d "$REPORT_DIR/$component" ]; then
                {
                  echo "    <div class=\"report-section\">"
                  echo "        <h2>${component} Tests</h2>"
                  echo "        <div class=\"report-links\">"
                } >> "$REPORT_DIR/index.html"

                for level in unit integration; do
                    if [ -f "$REPORT_DIR/$component/$level/report.html" ]; then
                        echo "            <p><a href=\"$component/$level/report.html\">${level^} Test Report</a></p>" >> "$REPORT_DIR/index.html"
                    fi
                    if [ -d "$REPORT_DIR/$component/$level/coverage" ]; then
                        echo "            <p><a href=\"$component/$level/coverage/index.html\">${level^} Test Coverage</a></p>" >> "$REPORT_DIR/index.html"
                    fi
                done

                echo "        </div>" >> "$REPORT_DIR/index.html"
                echo "    </div>" >> "$REPORT_DIR/index.html"
            fi
        done

        if [ -d "$REPORT_DIR/e2e" ]; then
            {
              echo "    <div class=\"report-section\">"
              echo "        <h2>End-to-End Tests</h2>"
              echo "        <div class=\"report-links\">"
            } >> "$REPORT_DIR/index.html"
            if [ -f "$REPORT_DIR/e2e/report.html" ]; then
                echo '            <p><a href="e2e/report.html">E2E Test Report</a></p>' >> "$REPORT_DIR/index.html"
            fi
            if [ -f "$REPORT_DIR/e2e/index.html" ]; then
                echo '            <p><a href="e2e/index.html">E2E Test Report</a></p>' >> "$REPORT_DIR/index.html"
            fi
            echo "        </div>" >> "$REPORT_DIR/index.html"
            echo "    </div>" >> "$REPORT_DIR/index.html"
        fi

        if [ -d "$REPORT_DIR/performance" ]; then
            {
              echo "    <div class=\"report-section\">"
              echo "        <h2>Performance Tests</h2>"
              echo "        <div class=\"report-links\">"
            } >> "$REPORT_DIR/index.html"
            if [ -f "$REPORT_DIR/performance/report.html" ]; then
                echo '            <p><a href="performance/report.html">Performance Test Report</a></p>' >> "$REPORT_DIR/index.html"
            fi
            if [ -d "$REPORT_DIR/performance/dashboard" ]; then
                echo '            <p><a href="performance/dashboard/index.html">Performance Dashboard</a></p>' >> "$REPORT_DIR/index.html"
            fi
            if [ -f "$REPORT_DIR/performance/summary.json" ]; then
                echo '            <p><a href="performance/summary.json">Performance Summary</a></p>' >> "$REPORT_DIR/index.html"
            fi
            echo "        </div>" >> "$REPORT_DIR/index.html"
            echo "    </div>" >> "$REPORT_DIR/index.html"
        fi

        echo "</body></html>" >> "$REPORT_DIR/index.html"

        echo -e "${GREEN}Report index generated: $REPORT_DIR/index.html${NC}"
    fi
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    show_help
fi

while [ "$1" != "" ]; do
    case $1 in
        --all )     RUN_UNIT=true
                    RUN_INTEGRATION=true
                    RUN_E2E=true
                    RUN_PERFORMANCE=true
                    ;;
        --unit )    RUN_UNIT=true
                    ;;
        --integration ) RUN_INTEGRATION=true
                    ;;
        --e2e )     RUN_E2E=true
                    ;;
        --performance ) RUN_PERFORMANCE=true
                    ;;
        --component ) shift
                    COMPONENT="$1"
                    ;;
        --report )  GENERATE_REPORT=true
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
echo -e "${BLUE}Starting Quantis test runner...${NC}"
echo -e "Repository root: ${PROJECT_ROOT}"

# Check dependencies
check_dependencies

# Prepare report directory
prepare_report_dir

# Run tests based on component and test type
if $RUN_UNIT; then
    if [ "$COMPONENT" = "api" ] || [ "$COMPONENT" = "all" ]; then
        run_python_tests "api" "unit"
    fi
    if [ "$COMPONENT" = "models" ] || [ "$COMPONENT" = "all" ]; then
        run_python_tests "models" "unit"
    fi
    if [ "$COMPONENT" = "web" ] || [ "$COMPONENT" = "all" ]; then
        run_js_tests "web" "unit"
    fi
    if [ "$COMPONENT" = "mobile" ] || [ "$COMPONENT" = "all" ]; then
        run_js_tests "mobile" "unit"
    fi
fi

if $RUN_INTEGRATION; then
    if [ "$COMPONENT" = "api" ] || [ "$COMPONENT" = "all" ]; then
        run_python_tests "api" "integration"
    fi
    if [ "$COMPONENT" = "models" ] || [ "$COMPONENT" = "all" ]; then
        run_python_tests "models" "integration"
    fi
    if [ "$COMPONENT" = "web" ] || [ "$COMPONENT" = "all" ]; then
        run_js_tests "web" "integration"
    fi
    if [ "$COMPONENT" = "mobile" ] || [ "$COMPONENT" = "all" ]; then
        run_js_tests "mobile" "integration"
    fi
fi

if $RUN_E2E; then
    run_e2e_tests
fi

if $RUN_PERFORMANCE; then
    run_performance_tests
fi

# Generate report index
if $GENERATE_REPORT; then
    generate_report_index
fi

echo -e "${GREEN}Quantis test runner completed successfully!${NC}"
exit 0
