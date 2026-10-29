#!/bin/bash
# linting.sh - Comprehensive linting script for Quantis project
#
# This script provides linting capabilities for all components of the Quantis project:
# - Python code (flake8, pylint, mypy)
# - JavaScript/TypeScript (ESLint, Prettier)
# - Shell scripts (shellcheck)
# - YAML files (yamllint)
# - Markdown files (markdownlint)
#
# Usage: ./linting.sh [options]
# Options:
#   --all                Lint all components
#   --python             Lint only Python code
#   --js                 Lint only JavaScript/TypeScript code
#   --shell              Lint only shell scripts
#   --yaml               Lint only YAML files
#   --markdown           Lint only Markdown files
#   --fix                Attempt to automatically fix issues
#   --report             Generate HTML report of linting results
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
LINT_PYTHON=false
LINT_JS=false
LINT_SHELL=false
LINT_YAML=false
LINT_MARKDOWN=false
LINT_ALL=false
AUTO_FIX=false
GENERATE_REPORT=false
# Resolve the actual repository root regardless of the caller's current
# directory - using $(pwd) here meant running this script from inside
# scripts/ (a very natural way to invoke it) silently limited every "find"
# below to the scripts/ directory itself, reporting "no files found" for
# every category.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPORT_DIR="$PROJECT_ROOT/lint_reports"

# Function to display help message
show_help() {
    echo -e "${BLUE}Linting Script for Quantis Project${NC}"
    echo ""
    echo "Usage: ./linting.sh [options]"
    echo ""
    echo "Options:"
    echo "  --all                Lint all components"
    echo "  --python             Lint only Python code"
    echo "  --js                 Lint only JavaScript/TypeScript code"
    echo "  --shell              Lint only shell scripts"
    echo "  --yaml               Lint only YAML files"
    echo "  --markdown           Lint only Markdown files"
    echo "  --fix                Attempt to automatically fix issues"
    echo "  --report             Generate HTML report of linting results"
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
    echo -e "${BLUE}Checking linting dependencies...${NC}"

    if $LINT_PYTHON || $LINT_ALL; then
        if ! command_exists flake8; then
            echo -e "${YELLOW}Warning: flake8 is not installed. Installing...${NC}"
            pip install flake8
        fi

        if ! command_exists pylint; then
            echo -e "${YELLOW}Warning: pylint is not installed. Installing...${NC}"
            pip install pylint
        fi

        if ! command_exists mypy; then
            echo -e "${YELLOW}Warning: mypy is not installed. Installing...${NC}"
            pip install mypy
        fi
    fi

    if $LINT_JS || $LINT_ALL; then
        if ! command_exists eslint; then
            echo -e "${YELLOW}Warning: eslint is not installed. Installing...${NC}"
            npm install -g eslint
        fi

        if ! command_exists prettier; then
            echo -e "${YELLOW}Warning: prettier is not installed. Installing...${NC}"
            npm install -g prettier
        fi
    fi

    if $LINT_SHELL || $LINT_ALL; then
        if ! command_exists shellcheck; then
            echo -e "${YELLOW}Warning: shellcheck is not installed. Installing...${NC}"
            if command_exists apt-get; then
                if sudo -n true 2>/dev/null; then
                    sudo apt-get update && sudo apt-get install -y shellcheck
                else
                    echo -e "${RED}Error: shellcheck is missing and passwordless sudo is not available.${NC}"
                    echo -e "${RED}Install it manually with: sudo apt-get install -y shellcheck${NC}"
                    exit 1
                fi
            elif command_exists brew; then
                brew install shellcheck
            else
                echo -e "${RED}Error: Cannot install shellcheck. Please install it manually.${NC}"
                exit 1
            fi
        fi
    fi

    if $LINT_YAML || $LINT_ALL; then
        if ! command_exists yamllint; then
            echo -e "${YELLOW}Warning: yamllint is not installed. Installing...${NC}"
            pip install yamllint
        fi
    fi

    if $LINT_MARKDOWN || $LINT_ALL; then
        if ! command_exists markdownlint; then
            echo -e "${YELLOW}Warning: markdownlint is not installed. Installing...${NC}"
            npm install -g markdownlint-cli
        fi
    fi

    echo -e "${GREEN}All required linting dependencies are installed.${NC}"
}

# Function to prepare report directory
prepare_report_dir() {
    if $GENERATE_REPORT; then
        echo -e "${BLUE}Preparing report directory...${NC}"
        mkdir -p "$REPORT_DIR"
        echo -e "${GREEN}Report directory prepared.${NC}"
    fi
}

# Function to lint Python code
lint_python() {
    echo -e "${BLUE}Linting Python code...${NC}"

    # Find all Python files
    mapfile -t PYTHON_FILES < <(find "$PROJECT_ROOT" -type f -name "*.py" | grep -v "/venv/" | grep -v "__pycache__/" | grep -v "/\.git/")

    if [ "${#PYTHON_FILES[@]}" -eq 0 ]; then
        echo -e "${YELLOW}No Python files found to lint.${NC}"
        return
    fi

    # Run flake8
    echo "Running flake8..."
    if $GENERATE_REPORT; then
        flake8 "${PYTHON_FILES[@]}" --output-file="$REPORT_DIR/flake8_report.txt" || true
    else
        flake8 "${PYTHON_FILES[@]}" || true
    fi

    # Run pylint
    echo "Running pylint..."
    if $GENERATE_REPORT; then
        pylint "${PYTHON_FILES[@]}" --output-format=text > "$REPORT_DIR/pylint_report.txt" || true
    else
        pylint "${PYTHON_FILES[@]}" || true
    fi

    # Run mypy
    echo "Running mypy..."
    if $GENERATE_REPORT; then
        mypy "${PYTHON_FILES[@]}" --no-error-summary > "$REPORT_DIR/mypy_report.txt" 2>&1 || true
    else
        mypy "${PYTHON_FILES[@]}" || true
    fi

    echo -e "${GREEN}Python linting completed.${NC}"
}

# Function to lint JavaScript/TypeScript code
lint_js() {
    echo -e "${BLUE}Linting JavaScript/TypeScript code...${NC}"

    # Check if an ESLint config exists. This project uses per-directory
    # configs (web-frontend/.eslintrc.cjs, mobile-frontend/.eslintrc.cjs)
    # rather than a single root-level one, so check broadly instead of
    # assuming a root .eslintrc.js/.json that will never exist here.
    if ! find "$PROJECT_ROOT" -maxdepth 2 \( -name ".eslintrc*" \) -not -path "*/node_modules/*" | grep -q .; then
        echo -e "${YELLOW}Warning: No ESLint configuration found. Using default configuration.${NC}"
    fi

    # Find all JS/TS files
    mapfile -t JS_FILES < <(find "$PROJECT_ROOT" -type f \( -name "*.js" -o -name "*.jsx" -o -name "*.ts" -o -name "*.tsx" \) | grep -v "/node_modules/" | grep -v "/\.git/")

    if [ "${#JS_FILES[@]}" -eq 0 ]; then
        echo -e "${YELLOW}No JavaScript/TypeScript files found to lint.${NC}"
        return
    fi

    # Run ESLint
    echo "Running ESLint..."
    if $AUTO_FIX; then
        if $GENERATE_REPORT; then
            eslint --fix "${JS_FILES[@]}" -o "$REPORT_DIR/eslint_report.json" --format json || true
        else
            eslint --fix "${JS_FILES[@]}" || true
        fi
    else
        if $GENERATE_REPORT; then
            eslint "${JS_FILES[@]}" -o "$REPORT_DIR/eslint_report.json" --format json || true
        else
            eslint "${JS_FILES[@]}" || true
        fi
    fi

    # Run Prettier
    echo "Running Prettier..."
    if $AUTO_FIX; then
        prettier --write "${JS_FILES[@]}"
    else
        if $GENERATE_REPORT; then
            prettier --check "${JS_FILES[@]}" > "$REPORT_DIR/prettier_report.txt" 2>&1 || true
        else
            prettier --check "${JS_FILES[@]}" || true
        fi
    fi

    echo -e "${GREEN}JavaScript/TypeScript linting completed.${NC}"
}

# Function to lint shell scripts
lint_shell() {
    echo -e "${BLUE}Linting shell scripts...${NC}"

    # Find all shell scripts
    mapfile -t SHELL_FILES < <(find "$PROJECT_ROOT" -type f -name "*.sh" | grep -v "/\.git/")

    if [ "${#SHELL_FILES[@]}" -eq 0 ]; then
        echo -e "${YELLOW}No shell scripts found to lint.${NC}"
        return
    fi

    # Run shellcheck
    echo "Running shellcheck..."
    if $GENERATE_REPORT; then
        shellcheck -f checkstyle "${SHELL_FILES[@]}" > "$REPORT_DIR/shellcheck_report.xml" || true
    else
        shellcheck "${SHELL_FILES[@]}" || true
    fi

    echo -e "${GREEN}Shell script linting completed.${NC}"
}

# Function to lint YAML files
lint_yaml() {
    echo -e "${BLUE}Linting YAML files...${NC}"

    # Find all YAML files
    mapfile -t YAML_FILES < <(find "$PROJECT_ROOT" -type f \( -name "*.yml" -o -name "*.yaml" \) | grep -v "/\.git/")

    if [ "${#YAML_FILES[@]}" -eq 0 ]; then
        echo -e "${YELLOW}No YAML files found to lint.${NC}"
        return
    fi

    # Run yamllint
    echo "Running yamllint..."
    if $GENERATE_REPORT; then
        yamllint -f parsable "${YAML_FILES[@]}" > "$REPORT_DIR/yamllint_report.txt" || true
    else
        yamllint "${YAML_FILES[@]}" || true
    fi

    echo -e "${GREEN}YAML linting completed.${NC}"
}

# Function to lint Markdown files
lint_markdown() {
    echo -e "${BLUE}Linting Markdown files...${NC}"

    # Find all Markdown files
    mapfile -t MD_FILES < <(find "$PROJECT_ROOT" -type f -name "*.md" | grep -v "/\.git/")

    if [ "${#MD_FILES[@]}" -eq 0 ]; then
        echo -e "${YELLOW}No Markdown files found to lint.${NC}"
        return
    fi

    # Run markdownlint
    echo "Running markdownlint..."
    if $GENERATE_REPORT; then
        markdownlint "${MD_FILES[@]}" > "$REPORT_DIR/markdownlint_report.txt" || true
    else
        markdownlint "${MD_FILES[@]}" || true
    fi

    echo -e "${GREEN}Markdown linting completed.${NC}"
}

# Function to generate HTML report
generate_html_report() {
    if $GENERATE_REPORT; then
        echo -e "${BLUE}Generating HTML report...${NC}"

        # Create HTML report
        HTML_REPORT="$REPORT_DIR/lint_report.html"

        # Create HTML header
        cat > "$HTML_REPORT" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantis Linting Report</title>
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
        .report-section {
            margin-bottom: 30px;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
        }
        .report-content {
            white-space: pre-wrap;
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        .summary {
            background-color: #e6f7ff;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .timestamp {
            color: #666;
            font-style: italic;
        }
    </style>
</head>
<body>
    <h1>Quantis Linting Report</h1>
    <div class="timestamp">Generated on $(date)</div>

    <div class="summary">
        <h2>Summary</h2>
        <p>This report contains the results of linting the Quantis codebase.</p>
    </div>
EOF

        # Add each report section
        if $LINT_PYTHON || $LINT_ALL; then
            echo -e "    <div class=\"report-section\">\n        <h2>Python Linting Results</h2>" >> "$HTML_REPORT"

            if [ -f "$REPORT_DIR/flake8_report.txt" ]; then
                echo -e "        <h3>Flake8</h3>\n        <div class=\"report-content\">" >> "$HTML_REPORT"
                cat "$REPORT_DIR/flake8_report.txt" >> "$HTML_REPORT"
                echo -e "        </div>" >> "$HTML_REPORT"
            fi

            if [ -f "$REPORT_DIR/pylint_report.txt" ]; then
                echo -e "        <h3>Pylint</h3>\n        <div class=\"report-content\">" >> "$HTML_REPORT"
                cat "$REPORT_DIR/pylint_report.txt" >> "$HTML_REPORT"
                echo -e "        </div>" >> "$HTML_REPORT"
            fi

            if [ -f "$REPORT_DIR/mypy_report.txt" ]; then
                echo -e "        <h3>Mypy</h3>\n        <div class=\"report-content\">" >> "$HTML_REPORT"
                cat "$REPORT_DIR/mypy_report.txt" >> "$HTML_REPORT"
                echo -e "        </div>" >> "$HTML_REPORT"
            fi

            echo -e "    </div>" >> "$HTML_REPORT"
        fi

        if $LINT_JS || $LINT_ALL; then
            echo -e "    <div class=\"report-section\">\n        <h2>JavaScript/TypeScript Linting Results</h2>" >> "$HTML_REPORT"

            if [ -f "$REPORT_DIR/eslint_report.json" ]; then
                echo -e "        <h3>ESLint</h3>\n        <div class=\"report-content\">" >> "$HTML_REPORT"
                cat "$REPORT_DIR/eslint_report.json" >> "$HTML_REPORT"
                echo -e "        </div>" >> "$HTML_REPORT"
            fi

            if [ -f "$REPORT_DIR/prettier_report.txt" ]; then
                echo -e "        <h3>Prettier</h3>\n        <div class=\"report-content\">" >> "$HTML_REPORT"
                cat "$REPORT_DIR/prettier_report.txt" >> "$HTML_REPORT"
                echo -e "        </div>" >> "$HTML_REPORT"
            fi

            echo -e "    </div>" >> "$HTML_REPORT"
        fi

        if $LINT_SHELL || $LINT_ALL; then
            if [ -f "$REPORT_DIR/shellcheck_report.xml" ]; then
                echo -e "    <div class=\"report-section\">\n        <h2>Shell Script Linting Results</h2>" >> "$HTML_REPORT"
                echo -e "        <h3>ShellCheck</h3>\n        <div class=\"report-content\">" >> "$HTML_REPORT"
                cat "$REPORT_DIR/shellcheck_report.xml" >> "$HTML_REPORT"
                echo -e "        </div>\n    </div>" >> "$HTML_REPORT"
            fi
        fi

        if $LINT_YAML || $LINT_ALL; then
            if [ -f "$REPORT_DIR/yamllint_report.txt" ]; then
                echo -e "    <div class=\"report-section\">\n        <h2>YAML Linting Results</h2>" >> "$HTML_REPORT"
                echo -e "        <h3>YAMLLint</h3>\n        <div class=\"report-content\">" >> "$HTML_REPORT"
                cat "$REPORT_DIR/yamllint_report.txt" >> "$HTML_REPORT"
                echo -e "        </div>\n    </div>" >> "$HTML_REPORT"
            fi
        fi

        if $LINT_MARKDOWN || $LINT_ALL; then
            if [ -f "$REPORT_DIR/markdownlint_report.txt" ]; then
                echo -e "    <div class=\"report-section\">\n        <h2>Markdown Linting Results</h2>" >> "$HTML_REPORT"
                echo -e "        <h3>MarkdownLint</h3>\n        <div class=\"report-content\">" >> "$HTML_REPORT"
                cat "$REPORT_DIR/markdownlint_report.txt" >> "$HTML_REPORT"
                echo -e "        </div>\n    </div>" >> "$HTML_REPORT"
            fi
        fi

        # Close HTML
        echo -e "</body>\n</html>" >> "$HTML_REPORT"

        echo -e "${GREEN}HTML report generated: $HTML_REPORT${NC}"
    fi
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    show_help
fi

while [ "$1" != "" ]; do
    case $1 in
        --all )     LINT_PYTHON=true
                    LINT_JS=true
                    LINT_SHELL=true
                    LINT_YAML=true
                    LINT_MARKDOWN=true
                    LINT_ALL=true
                    ;;
        --python )  LINT_PYTHON=true
                    ;;
        --js )      LINT_JS=true
                    ;;
        --shell )   LINT_SHELL=true
                    ;;
        --yaml )    LINT_YAML=true
                    ;;
        --markdown ) LINT_MARKDOWN=true
                    ;;
        --fix )     AUTO_FIX=true
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
echo -e "${BLUE}Starting Quantis linting process...${NC}"

# Check dependencies
check_dependencies

# Prepare report directory
prepare_report_dir

# Lint components
if $LINT_PYTHON || $LINT_ALL; then
    lint_python
fi

if $LINT_JS || $LINT_ALL; then
    lint_js
fi

if $LINT_SHELL || $LINT_ALL; then
    lint_shell
fi

if $LINT_YAML || $LINT_ALL; then
    lint_yaml
fi

if $LINT_MARKDOWN || $LINT_ALL; then
    lint_markdown
fi

# Generate HTML report
if $GENERATE_REPORT; then
    generate_html_report
fi

echo -e "${GREEN}Quantis linting process completed successfully!${NC}"
exit 0
