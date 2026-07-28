#!/bin/bash

# Linting and Fixing Script for Quantis Project (Python, JavaScript/TypeScript, YAML)

set -uo pipefail  # Don't hard-exit on the first lint failure — we want to run every tool and report a summary.

# Always operate relative to the actual repository root, regardless of
# where this script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}" || { echo "Failed to reach repository root"; exit 1; }

echo "----------------------------------------"
echo "Starting linting and fixing process for Quantis..."
echo "Repository root: ${PROJECT_ROOT}"
echo "----------------------------------------"

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check for required tools
echo "Checking for required tools..."

if ! command_exists python3; then
  echo "Error: python3 is required but not installed. Please install Python 3."
  exit 1
else
  echo "python3 is installed."
fi

if ! command_exists pip3; then
  echo "Error: pip3 is required but not installed. Please install pip3."
  exit 1
else
  echo "pip3 is installed."
fi

if ! command_exists node; then
  echo "Error: node is required but not installed. Please install Node.js."
  exit 1
else
  echo "node is installed."
fi

if ! command_exists npm; then
  echo "Error: npm is required but not installed. Please install npm."
  exit 1
else
  echo "npm is installed."
fi

if ! command_exists yamllint; then
  echo "Warning: yamllint is not installed. YAML validation will be limited."
  YAMLLINT_AVAILABLE=false
else
  echo "yamllint is installed."
  YAMLLINT_AVAILABLE=true
fi

# Install required Python linting tools if not already installed
echo "----------------------------------------"
echo "Installing/Updating Python linting tools..."
pip3 install --upgrade --quiet black isort flake8 pylint

# Define directories to process — these match the actual repository layout
# (code/backend/* for the API, code/quant_ml for the ML library).
PYTHON_DIRECTORIES=(
  "code/backend"
  "code/backend/auth"
  "code/backend/core"
  "code/backend/domain"
  "code/backend/endpoints"
  "code/backend/middleware"
  "code/backend/services"
  "code/backend/workers"
  "code/backend/tests"
  "code/quant_ml"
)

# ESLint and Prettier lint recursively by default, so we only need each
# frontend's top-level src/ directory rather than every nested subfolder
# (which was previously listed by hand and drifted out of sync with the
# real, evolving directory structure).
JS_PROJECT_DIRECTORIES=(
  "web-frontend"
  "mobile-frontend"
)

YAML_DIRECTORIES=(
  "infrastructure"
  "infrastructure/kubernetes"
  "infrastructure/ansible"
  "infrastructure/monitoring"
  ".github/workflows"
)

LINT_FAILURES=0

# 1. Python Linting
echo "----------------------------------------"
echo "Running Python linting tools..."

echo "Running Black code formatter..."
for dir in "${PYTHON_DIRECTORIES[@]}"; do
  if [ -d "${PROJECT_ROOT}/${dir}" ]; then
    echo "Formatting Python files in $dir..."
    python3 -m black "${PROJECT_ROOT}/${dir}" || {
      echo "Black encountered issues in $dir. Please review the above errors."
      LINT_FAILURES=$((LINT_FAILURES + 1))
    }
  else
    echo "Directory $dir not found. Skipping Black formatting for this directory."
  fi
done
echo "Black formatting completed."

echo "Running isort to sort imports..."
for dir in "${PYTHON_DIRECTORIES[@]}"; do
  if [ -d "${PROJECT_ROOT}/${dir}" ]; then
    echo "Sorting imports in Python files in $dir..."
    python3 -m isort "${PROJECT_ROOT}/${dir}" || {
      echo "isort encountered issues in $dir. Please review the above errors."
      LINT_FAILURES=$((LINT_FAILURES + 1))
    }
  else
    echo "Directory $dir not found. Skipping isort for this directory."
  fi
done
echo "Import sorting completed."

echo "Running flake8 linter..."
for dir in "${PYTHON_DIRECTORIES[@]}"; do
  if [ -d "${PROJECT_ROOT}/${dir}" ]; then
    echo "Linting Python files in $dir with flake8..."
    python3 -m flake8 "${PROJECT_ROOT}/${dir}" || {
      echo "Flake8 found issues in $dir. Please review the above warnings/errors."
      LINT_FAILURES=$((LINT_FAILURES + 1))
    }
  else
    echo "Directory $dir not found. Skipping flake8 for this directory."
  fi
done
echo "Flake8 linting completed."

echo "Running pylint for more comprehensive linting..."
for dir in "${PYTHON_DIRECTORIES[@]}"; do
  if [ -d "${PROJECT_ROOT}/${dir}" ]; then
    echo "Linting Python files in $dir with pylint..."
    py_files=$(find "${PROJECT_ROOT}/${dir}" -maxdepth 1 -type f -name "*.py")
    if [ -n "${py_files}" ]; then
      # shellcheck disable=SC2086
      python3 -m pylint --disable=C0111,C0103,C0303,W0621,C0301,W0612,W0611,R0913,R0914,R0915 ${py_files} || {
        echo "Pylint found issues in $dir. Please review the above warnings/errors."
        LINT_FAILURES=$((LINT_FAILURES + 1))
      }
    else
      echo "No top-level .py files in $dir. Skipping pylint for this directory."
    fi
  else
    echo "Directory $dir not found. Skipping pylint for this directory."
  fi
done
echo "Pylint linting completed."

# 2. JavaScript/TypeScript Linting
echo "----------------------------------------"
echo "Running JavaScript/TypeScript linting tools..."
echo "(web-frontend and mobile-frontend each already ship their own"
echo " project-scoped ESLint config with 'root: true', so no shared"
echo " top-level .eslintrc is generated here.)"

for project in "${JS_PROJECT_DIRECTORIES[@]}"; do
  project_path="${PROJECT_ROOT}/${project}"
  if [ ! -d "${project_path}" ]; then
    echo "Directory $project not found. Skipping JS/TS linting for this project."
    continue
  fi
  if [ ! -f "${project_path}/package.json" ]; then
    echo "package.json not found in $project. Skipping JS/TS linting for this project."
    continue
  fi
  if [ ! -d "${project_path}/node_modules" ]; then
    echo "node_modules not found in $project — installing dependencies first..."
    (cd "${project_path}" && npm install --no-audit --no-fund) || {
      echo "Failed to install dependencies in $project. Skipping lint for this project."
      LINT_FAILURES=$((LINT_FAILURES + 1))
      continue
    }
  fi

  echo "Linting $project with its local ESLint config (--fix)..."
  (cd "${project_path}" && npx eslint src --ext .js,.jsx,.ts,.tsx --fix) || {
    echo "ESLint found issues in $project. Please review the above warnings/errors."
    LINT_FAILURES=$((LINT_FAILURES + 1))
  }

  echo "Formatting $project with Prettier..."
  (cd "${project_path}" && npx --yes prettier --write "src/**/*.{js,jsx,ts,tsx}") || {
    echo "Prettier encountered issues in $project. Please review the above errors."
    LINT_FAILURES=$((LINT_FAILURES + 1))
  }
done
echo "JavaScript/TypeScript linting completed."

# 3. YAML Linting
echo "----------------------------------------"
echo "Running YAML linting tools..."

if [ "$YAMLLINT_AVAILABLE" = true ]; then
  echo "Running yamllint for YAML files..."
  for dir in "${YAML_DIRECTORIES[@]}"; do
    target="${PROJECT_ROOT}/${dir}"
    if [ -d "${target}" ]; then
      echo "Linting YAML files in $dir with yamllint..."
      yamllint "${target}" || {
        echo "yamllint found issues in $dir. Please review the above warnings/errors."
        LINT_FAILURES=$((LINT_FAILURES + 1))
      }
    elif [ -f "${target}" ]; then
      echo "Linting YAML file $dir with yamllint..."
      yamllint "${target}" || {
        echo "yamllint found issues in $dir. Please review the above warnings/errors."
        LINT_FAILURES=$((LINT_FAILURES + 1))
      }
    else
      echo "Directory/File $dir not found. Skipping yamllint for this path."
    fi
  done
  echo "yamllint completed."
else
  echo "Skipping yamllint (not installed)."

  echo "Performing basic YAML validation using Python..."
  pip3 install --upgrade --quiet pyyaml

  YAML_FILES_TO_VALIDATE=()
  for dir in "${YAML_DIRECTORIES[@]}"; do
    target="${PROJECT_ROOT}/${dir}"
    if [ -d "${target}" ]; then
      while IFS= read -r -d $'\0' file; do
        YAML_FILES_TO_VALIDATE+=("$file")
      done < <(find "${target}" -type f \( -name "*.yaml" -o -name "*.yml" \) -print0)
    elif [ -f "${target}" ]; then
       YAML_FILES_TO_VALIDATE+=("${target}")
    fi
  done

  echo "Validating ${#YAML_FILES_TO_VALIDATE[@]} YAML file(s)..."
  for file in "${YAML_FILES_TO_VALIDATE[@]}"; do
      echo "Validating $file..."
      python3 -c "import sys, yaml; yaml.safe_load_all(open(sys.argv[1]))" "$file" || {
          echo "YAML validation found issues in $file. Please review the above errors."
          LINT_FAILURES=$((LINT_FAILURES + 1))
      }
  done
  echo "Basic YAML validation completed."
fi

# 4. Common Fixes for All File Types
echo "----------------------------------------"
echo "Applying common fixes to all file types..."

echo "Fixing trailing whitespace..."
find "${PROJECT_ROOT}" -type f \( -name "*.py" -o -name "*.js" -o -name "*.jsx" -o -name "*.ts" -o -name "*.tsx" -o -name "*.yaml" -o -name "*.yml" \) \
  -not -path "*/node_modules/*" -not -path "*/venv/*" -not -path "*/dist/*" -not -path "*/build/*" -not -path "*/.git/*" \
  -exec sed -i 's/[ \t]*$//' {} \;
echo "Fixed trailing whitespace."

echo "Ensuring newline at end of files..."
find "${PROJECT_ROOT}" -type f \( -name "*.py" -o -name "*.js" -o -name "*.jsx" -o -name "*.ts" -o -name "*.tsx" -o -name "*.yaml" -o -name "*.yml" \) \
  -not -path "*/node_modules/*" -not -path "*/venv/*" -not -path "*/dist/*" -not -path "*/build/*" -not -path "*/.git/*" \
  -exec sh -c '[ -n "$(tail -c1 "$1")" ] && echo "" >> "$1"' sh {} \;
echo "Ensured newline at end of files."

echo "----------------------------------------"
if [ "${LINT_FAILURES}" -eq 0 ]; then
  echo "Linting and fixing process for Quantis completed with no unresolved issues!"
else
  echo "Linting and fixing process for Quantis completed with ${LINT_FAILURES} tool(s) reporting issues."
  echo "Review the output above for details."
fi
echo "----------------------------------------"
exit 0
