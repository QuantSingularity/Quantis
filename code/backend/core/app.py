"""
FastAPI application with comprehensive backend features
"""

import json
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import structlog
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from ..auth.auth import AuditLogger, rate_limit
from ..domain.models import User
from ..domain.schemas import HealthCheck
from ..endpoints import auth, datasets, financial
from ..endpoints import models as models_endpoint
from ..endpoints import monitoring, notifications, prediction, users, websocket
from .config import Settings, get_settings
from .database import close_redis, get_db, get_redis, health_check, init_db

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger(__name__)
settings: Settings = get_settings()
REQUEST_COUNT = Counter(
    "quantis_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_DURATION = Histogram(
    "quantis_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"],
)
WEBSOCKET_CONNECTIONS = Counter(
    "quantis_websocket_connections_total",
    "Total number of WebSocket connections",
    ["endpoint"],
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect request metrics"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        endpoint = request.url.path
        method = request.method
        status_code = response.status_code
        REQUEST_COUNT.labels(
            method=method, endpoint=endpoint, status_code=status_code
        ).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
        response.headers["X-Response-Time"] = str(duration)
        return response


class AuditMiddleware(BaseHTTPMiddleware):
    """Audit logging for sensitive requests"""

    async def dispatch(self, request: Request, call_next):
        request_id = f"req_{int(time.time() * 1000000.0)}"
        request.state.request_id = request_id
        response = await call_next(request)
        if settings.logging.enable_audit_logging and request.method in [
            "POST",
            "PUT",
            "DELETE",
            "PATCH",
        ]:
            try:
                user: Optional[User] = getattr(request.state, "user", None)
                user_id = user.id if user else None
                # BUG FIX: original code leaked the DB session (no close).
                # Use the generator protocol correctly with try/finally.
                db_gen = get_db()
                db = next(db_gen)
                try:
                    AuditLogger.log_event(
                        db=db,
                        user_id=user_id,
                        action=f"{request.method.lower()}_{request.url.path.replace('/', '_')}",
                        resource_type="api_endpoint",
                        resource_name=request.url.path,
                        request=request,
                        status_code=response.status_code,
                    )
                finally:
                    try:
                        next(db_gen)
                    except StopIteration:
                        pass
            except Exception as e:
                logger.error("Audit logging failed", error=str(e))
        return response


class WebSocketManager:
    """Manage WebSocket connections"""

    def __init__(self) -> None:
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        self.user_connections: Dict[int, list] = {}

    async def connect(self, websocket: WebSocket, connection_id: str, user_id: int):
        await websocket.accept()
        self.active_connections.setdefault(connection_id, {})[str(user_id)] = websocket
        self.user_connections.setdefault(user_id, []).append(connection_id)
        WEBSOCKET_CONNECTIONS.labels(endpoint=connection_id).inc()
        logger.info("WebSocket connected", connection_id=connection_id, user_id=user_id)

    def disconnect(self, connection_id: str, user_id: int) -> Any:
        self.active_connections.get(connection_id, {}).pop(str(user_id), None)
        if not self.active_connections.get(connection_id):
            self.active_connections.pop(connection_id, None)
        if user_id in self.user_connections:
            self.user_connections[user_id].remove(connection_id)
            if not self.user_connections[user_id]:
                self.user_connections.pop(user_id)
        logger.info(
            "WebSocket disconnected", connection_id=connection_id, user_id=user_id
        )

    async def send_personal_message(self, message: str, user_id: int):
        for connection_id in self.user_connections.get(user_id, []):
            ws = self.active_connections.get(connection_id, {}).get(str(user_id))
            if ws:
                try:
                    await ws.send_text(message)
                except Exception as e:
                    logger.error("Failed to send WebSocket message", error=str(e))
                    self.disconnect(connection_id, user_id)

    async def broadcast(self, message: str, connection_id: str):
        for ws in self.active_connections.get(connection_id, {}).values():
            try:
                await ws.send_text(message)
            except Exception as e:
                logger.error("Failed to broadcast WebSocket message", error=str(e))


websocket_manager = WebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Quantis API...")
    try:
        init_db()
        logger.info("Database initialized successfully")
        if settings.celery_broker_url:
            logger.info("Background tasks enabled")
    except Exception as e:
        logger.error("Startup failed", error=str(e))
        raise
    yield
    logger.info("Shutting down Quantis API...")
    try:
        await close_redis()
        logger.info("Redis connection closed")
    except Exception as e:
        logger.warning("Shutdown error (non-fatal)", error=str(e))
    logger.info("Quantis API shutdown complete")


app = FastAPI(
    title="Quantis API",
    version="2.0.0",
    lifespan=lifespan,
    default_response_class=JSONResponse,
    openapi_url="/openapi.json" if settings.debug else None,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)


@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request, exc):
    logger.warning(
        "HTTP Exception",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
    )
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.warning("Validation Error", errors=exc.errors(), path=request.url.path)
    return await request_validation_exception_handler(request, exc)


app.add_middleware(MetricsMiddleware)
app.add_middleware(AuditMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.get("/health", response_model=HealthCheck, tags=["System"])
async def health_check_endpoint(db=Depends(get_db)):
    try:
        health_result = health_check()
        db_status = "ok" if health_result.get("database") else "error"
        redis_status = "ok"
        try:
            redis_client = await get_redis()
            if redis_client is not None:
                await redis_client.ping()
            else:
                redis_status = "not_configured"
        except Exception:
            redis_status = "error"
        return {
            "status": "ok",
            "database": db_status,
            "redis": redis_status,
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail=f"Service Unavailable: {str(e)}")


@app.get("/metrics", tags=["System"])
async def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/info", tags=["System"])
@rate_limit(requests=10, window=60)
async def system_info():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "features": {
            "websockets": True,
            "background_tasks": bool(settings.celery_broker_url),
            "data_encryption": settings.compliance.enable_data_encryption,
            "data_masking": settings.compliance.enable_data_masking,
            "data_retention": settings.compliance.enable_data_retention_policies,
            "consent_management": settings.compliance.enable_consent_management,
            "audit_logging": settings.logging.enable_audit_logging,
        },
        "timestamp": time.time(),
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Welcome to Quantis API",
        "version": settings.app_version,
        "docs_url": "/docs" if settings.debug else None,
        "health_url": "/health",
        "metrics_url": "/metrics",
    }


app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/users", tags=["Users"])
app.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
app.include_router(models_endpoint.router, prefix="/models", tags=["Models"])
app.include_router(prediction.router, prefix="/predictions", tags=["Predictions"])
app.include_router(
    notifications.router, prefix="/notifications", tags=["Notifications"]
)
app.include_router(monitoring.router, prefix="/monitoring", tags=["Monitoring"])
app.include_router(financial.router, prefix="/financial", tags=["Financial"])
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])


@app.websocket("/ws/notifications/{user_id}")
async def websocket_notifications(websocket: WebSocket, user_id: int):
    try:
        await websocket_manager.connect(websocket, "notifications", user_id)
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            logger.info("Received WebSocket message", user_id=user_id, message=message)
            await websocket.send_text(
                f"Message received: {message.get('text', 'No text')}"
            )
    except WebSocketDisconnect:
        websocket_manager.disconnect("notifications", user_id)
        logger.info("WebSocket disconnected", user_id=user_id)
    except Exception as e:
        logger.error("WebSocket error", user_id=user_id, error=str(e))
        websocket_manager.disconnect("notifications", user_id)
