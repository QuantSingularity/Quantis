# Quantis Backend

## Quick Start

### Installation

```bash
# Extract the ZIP file
unzip code_deliverable.zip
cd code_fixed/api

# Install dependencies
pip install -r requirements.txt
```

### Starting the Backend

```bash
# From code_fixed directory
cd code_fixed
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

# With auto-reload for development
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Verification

The backend should start successfully with output:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Database initialized successfully
INFO:     Application startup complete.
```

## Architecture

```
code/
├── api/                    # Main backend application
│   ├── endpoints/          # API route handlers
│   ├── middleware/         # Request/response middleware
│   ├── services/           # Business logic layer
│   ├── models.py           # Database models
│   ├── schemas.py          # Pydantic validation schemas
│   ├── config.py           # Configuration management
│   ├── database.py         # Database connection/session
│   └── app.py              # FastAPI application entry
├── data/                   # Data processing modules
├── models/                 # ML model training/serving
├── notebooks/              # Jupyter notebooks
└── scripts/                # Utility scripts
```

## API Documentation

Once running, access:

- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **System Info**: http://localhost:8000/info

## Database

The backend uses SQLite by default (`sqlite:///./quantis.db`).
Database is automatically initialized on startup.

To use PostgreSQL instead, set environment variable:

```bash
export DATABASE_URL="postgresql://user:password@localhost/quantis"
```

## Environment Configuration

Create a `.env` file in `code/api/` directory:

```env
# Security
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET=your-jwt-secret-change-in-production

# Database (optional - defaults to SQLite)
DATABASE_URL=sqlite:///./quantis.db

# Redis (optional)
REDIS_URL=redis://localhost:6379/0

# Email (optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@example.com
SMTP_PASSWORD=your-password
```

## Testing

```bash
# Run tests (if available)
pytest api/

# Test import
python -c "from api import app; print('✓ Success')"

# Test server startup
timeout 5 python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```
