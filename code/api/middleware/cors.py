"""
CORS middleware for handling cross-origin requests
"""

from typing import Any, List, Optional

from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware


class CORSMiddleware(BaseHTTPMiddleware):
    """CORS middleware with additional security features"""

    def __init__(
        self,
        app: Any,
        allow_origins: Optional[List[str]] = None,
        allow_credentials: bool = True,
        allow_methods: Optional[List[str]] = None,
        allow_headers: Optional[List[str]] = None,
        expose_headers: Optional[List[str]] = None,
        max_age: int = 600,
        allow_origin_regex: Optional[str] = None,
    ) -> None:
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_credentials = allow_credentials
        self.allow_methods = allow_methods or [
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "PATCH",
            "OPTIONS",
            "HEAD",
        ]
        self.allow_headers = allow_headers or [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-API-Key",
            "X-Requested-With",
            "X-CSRF-Token",
        ]
        self.expose_headers = expose_headers or [
            "X-Process-Time",
            "X-Request-ID",
            "X-Rate-Limit-Remaining",
            "X-Rate-Limit-Reset",
        ]
        self.max_age = max_age
        self.allow_origin_regex = allow_origin_regex

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return self._create_preflight_response(request)
        response = await call_next(request)
        self._add_cors_headers(request, response)
        return response

    def _create_preflight_response(self, request: Request) -> Any:
        """Create response for preflight OPTIONS requests"""
        from fastapi.responses import Response

        response = Response()
        self._add_cors_headers(request, response)
        return response

    def _add_cors_headers(self, request: Request, response: Any) -> Any:
        """Add CORS headers to response"""
        origin = request.headers.get("origin")
        if self._is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
        response.headers["Access-Control-Expose-Headers"] = ", ".join(
            self.expose_headers
        )
        response.headers["Access-Control-Max-Age"] = str(self.max_age)

    def _is_origin_allowed(self, origin: Optional[str]) -> bool:
        """Check if the origin is allowed"""
        if not origin:
            return True
        if "*" in self.allow_origins:
            return True
        if origin in self.allow_origins:
            return True
        if self.allow_origin_regex:
            import re

            return bool(re.match(self.allow_origin_regex, origin))
        return False


def setup_cors(app: Any, environment: str = "development") -> Any:
    """Setup CORS middleware based on environment"""
    if environment == "production":
        allowed_origins = [
            "https://yourdomain.com",
            "https://www.yourdomain.com",
            "https://app.yourdomain.com",
        ]
        allow_credentials = True
    elif environment == "staging":
        allowed_origins = [
            "https://staging.yourdomain.com",
            "https://dev.yourdomain.com",
            "http://localhost:3000",
            "http://localhost:3001",
        ]
        allow_credentials = True
    else:
        allowed_origins = ["*"]
        allow_credentials = False
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=[
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-API-Key",
            "X-Requested-With",
            "X-CSRF-Token",
        ],
        expose_headers=[
            "X-Process-Time",
            "X-Request-ID",
            "X-Rate-Limit-Remaining",
            "X-Rate-Limit-Reset",
        ],
    )


def get_cors_config(environment: str = "development") -> dict:
    """Get CORS configuration for the given environment"""
    configs = {
        "development": {
            "allow_origins": ["*"],
            "allow_credentials": False,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
            "allow_headers": ["*"],
            "max_age": 600,
        },
        "staging": {
            "allow_origins": [
                "https://staging.yourdomain.com",
                "https://dev.yourdomain.com",
                "http://localhost:3000",
                "http://localhost:3001",
            ],
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
            "allow_headers": [
                "Accept",
                "Accept-Language",
                "Content-Language",
                "Content-Type",
                "Authorization",
                "X-API-Key",
                "X-Requested-With",
            ],
            "max_age": 3600,
        },
        "production": {
            "allow_origins": [
                "https://yourdomain.com",
                "https://www.yourdomain.com",
                "https://app.yourdomain.com",
            ],
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH"],
            "allow_headers": ["Accept", "Content-Type", "Authorization", "X-API-Key"],
            "max_age": 86400,
        },
    }
    return configs.get(environment, configs["development"])
