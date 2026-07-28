#!/bin/bash
# make_scripts_executable.sh - Make all automation scripts executable
#
# This script makes all .sh files in this scripts/ directory executable,
# regardless of the directory it's invoked from.
#
# Author: Abrar Ahmed
# Date: May 22, 2025

set -uo pipefail

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Always operate on this script's own directory, not the caller's current
# working directory — otherwise "find ." would scan (and chmod) unrelated
# shell scripts anywhere under wherever this happened to be invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}Making all automation scripts in ${SCRIPT_DIR} executable...${NC}"

FAILURES=0
COUNT=0

# Use a null-delimited find + read loop rather than word-splitting a
# command substitution, so filenames with spaces or special characters
# are handled correctly.
while IFS= read -r -d '' script; do
    if chmod +x "$script"; then
        echo "Made executable: $script"
        COUNT=$((COUNT + 1))
    else
        echo -e "${RED}Failed to chmod: $script${NC}"
        FAILURES=$((FAILURES + 1))
    fi
done < <(find "${SCRIPT_DIR}" -maxdepth 1 -name "*.sh" -print0)

if [ "${FAILURES}" -eq 0 ]; then
    echo -e "${GREEN}All ${COUNT} scripts are now executable!${NC}"
else
    echo -e "${RED}${FAILURES} script(s) could not be made executable.${NC}"
    exit 1
fi
