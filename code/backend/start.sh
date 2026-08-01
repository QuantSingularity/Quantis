#!/bin/bash
# Quantis API Startup Script

# Set working directory to script location
cd "$(dirname "$0")"

echo "================================================"
echo "Starting Quantis API"
echo "================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.installed" ]; then
    echo "Installing dependencies..."
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    touch venv/.installed
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "⚠️  WARNING: Using default development credentials!"
    echo "⚠️  Please edit .env before using in production!"
fi

# Initialize database
# NOTE: the backend package uses relative imports (e.g. `from ..auth.auth import ...`)
# so it must be imported as the `backend` package from its parent directory (`code/`),
# not as a standalone `api` package from inside `backend/`.
echo "Initializing database..."
python3 -c "
import sys
sys.path.insert(0, '..')
from backend.core.database import init_db
try:
    init_db()
    print('✓ Database initialized')
except Exception as e:
    print(f'✗ Database initialization failed: {e}')
    sys.exit(1)
"

# Start the API server
echo "Starting server on http://0.0.0.0:8000"
echo "API documentation will be available at http://localhost:8000/docs"
echo "================================================"

cd ..
uvicorn backend.core.app:app --host 0.0.0.0 --port 8000 --reload
