"""
Error handling middleware for comprehensive error management
"""

import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware to handle all exceptions and provide consistent error responses"""

    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except HTTPException as exc:
            # Handle FastAPI HTTP exceptions
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": True,
                    "message": exc.detail,
                    "status_code": exc.status_code,
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": str(request.url),
                },
            )
        except ValueError as exc:
            # Handle validation errors
            logger.error(f"Validation error: {str(exc)}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": True,
                    "message": f"Validation error: {str(exc)}",
                    "status_code": 400,
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": str(request.url),
                },
            )
        except Exception as exc:
            # Handle all other exceptions
            logger.error(f"Unhandled exception: {str(exc)}")
            logger.error(traceback.format_exc())

            return JSONResponse(
                status_code=500,
                content={
                    "error": True,
                    "message": "Internal server error",
                    "status_code": 500,
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": str(request.url),
                    "details": str(exc) if logger.level <= logging.DEBUG else None,
                },
            )


def create_error_response(
    status_code: int, message: str, details: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """Create a standardized error response"""
    content = {
        "error": True,
        "message": message,
        "status_code": status_code,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if details:
        content["details"] = details

    return JSONResponse(status_code=status_code, content=content)


def create_success_response(
    data: Any = None, message: str = "Success"
) -> Dict[str, Any]:
    """Create a standardized success response"""
    response = {
        "error": False,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if data is not None:
        response["data"] = data

    return response


# Custom exception classes
class ValidationError(Exception):
    """Custom validation error"""


class AuthenticationError(Exception):
    """Custom authentication error"""


class AuthorizationError(Exception):
    """Custom authorization error"""


class ResourceNotFoundError(Exception):
    """Custom resource not found error"""


class BusinessLogicError(Exception):
    """Custom business logic error"""


class ExternalServiceError(Exception):
    """Custom external service error"""
