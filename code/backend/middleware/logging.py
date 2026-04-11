"""
Logging middleware for request/response tracking and monitoring
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests and responses"""

    def __init__(
        self, app: Any, log_requests: bool = True, log_responses: bool = True
    ) -> None:
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        if self.log_requests:
            await self._log_request(request)
        response = await call_next(request)
        process_time = time.time() - start_time
        if self.log_responses:
            await self._log_response(request, response, process_time)
        response.headers["X-Process-Time"] = str(process_time)
        return response

    async def _log_request(self, request: Request):
        """Log incoming request details"""
        try:
            client_ip = request.client.host if request.client else "unknown"
            user_agent = request.headers.get("user-agent", "unknown")
            body = None
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    if len(body) > 1000:
                        body = body[:1000] + b"... (truncated)"
                    body = body.decode("utf-8", errors="ignore")
                except Exception:
                    body = "Could not read body"
            log_data = {
                "event": "request",
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "client_ip": client_ip,
                "user_agent": user_agent,
                "timestamp": datetime.utcnow().isoformat(),
                "body": body,
            }
            logger.info(f"Request: {request.method} {request.url.path}", extra=log_data)
        except Exception as e:
            logger.error(f"Error logging request: {str(e)}")

    async def _log_response(self, request: Request, response, process_time: float):
        """Log response details"""
        try:
            log_data = {
                "event": "response",
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "status_code": response.status_code,
                "process_time": process_time,
                "timestamp": datetime.utcnow().isoformat(),
            }
            if response.status_code >= 500:
                logger.error(
                    f"Response: {request.method} {request.url.path} - {response.status_code}",
                    extra=log_data,
                )
            elif response.status_code >= 400:
                logger.warning(
                    f"Response: {request.method} {request.url.path} - {response.status_code}",
                    extra=log_data,
                )
            else:
                logger.info(
                    f"Response: {request.method} {request.url.path} - {response.status_code}",
                    extra=log_data,
                )
        except Exception as e:
            logger.error(f"Error logging response: {str(e)}")


class MetricsCollector:
    """Simple metrics collector for monitoring"""

    def __init__(self) -> None:
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.endpoint_stats = {}
        self.status_code_stats = {}

    def record_request(
        self, method: str, path: str, status_code: int, response_time: float
    ) -> Any:
        """Record request metrics"""
        self.request_count += 1
        self.total_response_time += response_time
        if status_code >= 400:
            self.error_count += 1
        endpoint_key = f"{method} {path}"
        if endpoint_key not in self.endpoint_stats:
            self.endpoint_stats[endpoint_key] = {
                "count": 0,
                "total_time": 0.0,
                "errors": 0,
            }
        self.endpoint_stats[endpoint_key]["count"] += 1
        self.endpoint_stats[endpoint_key]["total_time"] += response_time
        if status_code >= 400:
            self.endpoint_stats[endpoint_key]["errors"] += 1
        if status_code not in self.status_code_stats:
            self.status_code_stats[status_code] = 0
        self.status_code_stats[status_code] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        avg_response_time = (
            self.total_response_time / self.request_count
            if self.request_count > 0
            else 0
        )
        error_rate = (
            self.error_count / self.request_count if self.request_count > 0 else 0
        )
        endpoint_metrics = {}
        for endpoint, stats in self.endpoint_stats.items():
            avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
            error_rate_endpoint = (
                stats["errors"] / stats["count"] if stats["count"] > 0 else 0
            )
            endpoint_metrics[endpoint] = {
                "count": stats["count"],
                "avg_response_time": avg_time,
                "error_rate": error_rate_endpoint,
            }
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": error_rate,
            "avg_response_time": avg_response_time,
            "status_codes": self.status_code_stats,
            "endpoints": endpoint_metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def reset_metrics(self) -> Any:
        """Reset all metrics"""
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.endpoint_stats = {}
        self.status_code_stats = {}


metrics_collector = MetricsCollector()


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect metrics"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        metrics_collector.record_request(
            request.method, request.url.path, response.status_code, process_time
        )
        return response
