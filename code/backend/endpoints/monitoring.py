"""
Monitoring and system health endpoints
"""

from datetime import datetime, timedelta
from typing import List, Optional

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from ..auth.auth import get_current_user, require_admin

# Local application imports - adjust if your project layout differs
from ..core.database import get_db
from ..domain.models import User
from ..services.user_service import UserService

router = APIRouter()


# -----------------------
# Pydantic response models
# -----------------------
class SystemHealth(BaseModel):
    status: str
    timestamp: str
    database_status: str
    api_status: str
    disk_usage: dict
    memory_usage: dict
    cpu_usage: float

    class Config:
        orm_mode = True


class SystemStats(BaseModel):
    total_users: int
    active_users: int
    total_datasets: int
    total_models: int
    trained_models: int
    total_predictions: int
    predictions_last_24h: int

    class Config:
        orm_mode = True


class AuditLogEntry(BaseModel):
    id: int
    user_id: Optional[int]
    username: Optional[str]
    action: str
    resource_type: str
    resource_id: Optional[str]
    details: Optional[dict]
    ip_address: Optional[str]
    created_at: str

    class Config:
        orm_mode = True


class MetricEntry(BaseModel):
    id: int
    metric_name: str
    metric_value: float
    metric_unit: Optional[str]
    tags: Optional[dict]
    created_at: str

    class Config:
        orm_mode = True


# -----------------------
# System health endpoints
# -----------------------
@router.get("/health", response_model=SystemHealth)
async def get_system_health(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get overall system health status."""
    try:
        # Test database connection
        db.execute(text("SELECT 1"))
        database_status = "healthy"
    except Exception:
        database_status = "unhealthy"

    # Get system metrics
    if PSUTIL_AVAILABLE:
        disk_usage = psutil.disk_usage("/")
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        disk_info = {
            "total": disk_usage.total,
            "used": disk_usage.used,
            "percent": disk_usage.percent,
        }
        mem_info = {
            "total": memory.total,
            "used": memory.used,
            "percent": memory.percent,
        }
        mem_percent = memory.percent
        disk_percent = disk_usage.percent
    else:
        disk_info = {"total": 0, "used": 0, "percent": 0}
        mem_info = {"total": 0, "used": 0, "percent": 0}
        mem_percent = 0
        disk_percent = 0
        cpu_percent = 0

    # Determine overall status
    overall_status = "healthy"
    if database_status == "unhealthy" or mem_percent > 90 or disk_percent > 90:
        overall_status = "unhealthy"
    elif mem_percent > 80 or disk_percent > 80 or cpu_percent > 80:
        overall_status = "warning"

    return SystemHealth(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        database_status=database_status,
        api_status="healthy",
        disk_usage=disk_info,
        memory_usage=mem_info,
        cpu_usage=cpu_percent,
    )


# --------------------------
# System usage statistics
# --------------------------
@router.get("/stats", response_model=SystemStats)
async def get_system_statistics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get system usage statistics."""

    # Count totals (only active users in total_users as the original did)
    total_users = db.query(models.User).filter(models.User.is_active.is_(True)).count()

    # Active users (logged in within last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    active_users = (
        db.query(models.User)
        .filter(
            models.User.is_active.is_(True),
            models.User.last_login >= thirty_days_ago,
        )
        .count()
    )

    total_datasets = (
        db.query(models.Dataset).filter(models.Dataset.is_active.is_(True)).count()
    )
    total_models = (
        db.query(models.Model).filter(models.Model.is_active.is_(True)).count()
    )
    trained_models = (
        db.query(models.Model)
        .filter(models.Model.is_active.is_(True), models.Model.status == "trained")
        .count()
    )

    total_predictions = db.query(models.Prediction).count()

    # Predictions in last 24 hours
    twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
    predictions_last_24h = (
        db.query(models.Prediction)
        .filter(models.Prediction.created_at >= twenty_four_hours_ago)
        .count()
    )

    return SystemStats(
        total_users=total_users,
        active_users=active_users,
        total_datasets=total_datasets,
        total_models=total_models,
        trained_models=trained_models,
        total_predictions=total_predictions,
        predictions_last_24h=predictions_last_24h,
    )


# -----------------------
# Audit logging endpoints
# -----------------------
@router.get("/audit-logs", response_model=List[AuditLogEntry])
async def get_audit_logs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    action: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    user_id: Optional[int] = Query(None),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Get audit logs (admin only)."""
    query = db.query(models.AuditLog)

    # Apply filters
    if action:
        query = query.filter(models.AuditLog.action == action)
    if resource_type:
        query = query.filter(models.AuditLog.resource_type == resource_type)
    if user_id:
        query = query.filter(models.AuditLog.user_id == user_id)

    # Get logs with user information
    logs = (
        query.order_by(models.AuditLog.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    user_service = UserService(db)

    result = []
    for log in logs:
        username = None
        if log.user_id:
            user = user_service.get_user_by_id(log.user_id)
            username = user.username if user else "Unknown"

        result.append(
            AuditLogEntry(
                id=log.id,
                user_id=log.user_id,
                username=username,
                action=log.action,
                resource_type=log.resource_type,
                resource_id=log.resource_id,
                details=log.details,
                ip_address=log.ip_address,
                created_at=log.created_at.isoformat(),
            )
        )

    return result


@router.post("/audit-logs")
async def create_audit_log(
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    details: Optional[dict] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create an audit log entry."""
    # Instantiate the ORM model
    audit_log = models.AuditLog(
        user_id=current_user.get("user_id"),
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
    )

    db.add(audit_log)
    db.commit()
    db.refresh(audit_log)

    return {"message": "Audit log created successfully", "id": audit_log.id}


# -----------------------
# System metrics endpoints
# -----------------------
@router.get("/metrics", response_model=List[MetricEntry])
async def get_system_metrics(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    metric_name: Optional[str] = Query(None),
    hours: int = Query(24, ge=1, le=168),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get system metrics."""
    query = db.query(models.SystemMetrics)

    # Filter by time range
    time_threshold = datetime.utcnow() - timedelta(hours=hours)
    query = query.filter(models.SystemMetrics.created_at >= time_threshold)

    # Filter by metric name if provided
    if metric_name:
        query = query.filter(models.SystemMetrics.metric_name == metric_name)

    metrics = (
        query.order_by(models.SystemMetrics.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return [
        MetricEntry(
            id=metric.id,
            metric_name=metric.metric_name,
            metric_value=metric.metric_value,
            metric_unit=metric.metric_unit,
            tags=metric.tags,
            created_at=metric.created_at.isoformat(),
        )
        for metric in metrics
    ]


@router.post("/metrics")
async def record_metric(
    metric_name: str,
    metric_value: float,
    metric_unit: Optional[str] = None,
    tags: Optional[dict] = None,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Record a system metric."""
    metric = models.SystemMetrics(
        metric_name=metric_name,
        metric_value=metric_value,
        metric_unit=metric_unit,
        tags=tags,
    )

    db.add(metric)
    db.commit()
    db.refresh(metric)

    return {"message": "Metric recorded successfully", "id": metric.id}


# -----------------------
# Performance analytics
# -----------------------
@router.get("/analytics/predictions")
async def get_prediction_analytics(
    days: int = Query(7, ge=1, le=90),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get prediction analytics over time."""
    time_threshold = datetime.utcnow() - timedelta(days=days)

    # Daily prediction counts
    daily_counts = (
        db.query(
            func.date(models.Prediction.created_at).label("date"),
            func.count(models.Prediction.id).label("count"),
        )
        .filter(models.Prediction.created_at >= time_threshold)
        .group_by(func.date(models.Prediction.created_at))
        .all()
    )

    # Average confidence by day
    daily_confidence = (
        db.query(
            func.date(models.Prediction.created_at).label("date"),
            func.avg(models.Prediction.confidence_score).label("avg_confidence"),
        )
        .filter(
            models.Prediction.created_at >= time_threshold,
            models.Prediction.confidence_score.isnot(None),
        )
        .group_by(func.date(models.Prediction.created_at))
        .all()
    )

    # Average execution time by day
    daily_execution_time = (
        db.query(
            func.date(models.Prediction.created_at).label("date"),
            func.avg(models.Prediction.execution_time_ms).label("avg_execution_time"),
        )
        .filter(
            models.Prediction.created_at >= time_threshold,
            models.Prediction.execution_time_ms.isnot(None),
        )
        .group_by(func.date(models.Prediction.created_at))
        .all()
    )

    return {
        "daily_prediction_counts": [
            {"date": str(row.date), "count": row.count} for row in daily_counts
        ],
        "daily_avg_confidence": [
            {"date": str(row.date), "avg_confidence": float(row.avg_confidence)}
            for row in daily_confidence
        ],
        "daily_avg_execution_time": [
            {
                "date": str(row.date),
                "avg_execution_time_ms": float(row.avg_execution_time),
            }
            for row in daily_execution_time
        ],
    }


@router.get("/analytics/models")
async def get_model_analytics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get model usage analytics."""
    # Model type distribution
    model_types = (
        db.query(models.Model.model_type, func.count(models.Model.id).label("count"))
        .filter(models.Model.is_active.is_(True))
        .group_by(models.Model.model_type)
        .all()
    )

    # Model status distribution
    model_status = (
        db.query(models.Model.status, func.count(models.Model.id).label("count"))
        .filter(models.Model.is_active.is_(True))
        .group_by(models.Model.status)
        .all()
    )

    # Most used models (by prediction count)
    popular_models = (
        db.query(
            models.Model.id,
            models.Model.name,
            func.count(models.Prediction.id).label("prediction_count"),
        )
        .join(models.Prediction, models.Model.id == models.Prediction.model_id)
        .filter(models.Model.is_active.is_(True))
        .group_by(models.Model.id, models.Model.name)
        .order_by(func.count(models.Prediction.id).desc())
        .limit(10)
        .all()
    )

    return {
        "model_type_distribution": [
            {"model_type": row.model_type, "count": row.count} for row in model_types
        ],
        "model_status_distribution": [
            {"status": row.status, "count": row.count} for row in model_status
        ],
        "most_popular_models": [
            {
                "model_id": row.id,
                "model_name": row.name,
                "prediction_count": row.prediction_count,
            }
            for row in popular_models
        ],
    }


# -----------------------
# System maintenance endpoints
# -----------------------
@router.post("/maintenance/cleanup")
async def cleanup_system(
    days_old: int = Query(30, ge=7, le=365),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Clean up old system data."""
    cutoff_date = datetime.utcnow() - timedelta(days=days_old)

    # Delete old audit logs and return number deleted
    audit_deleted = (
        db.query(models.AuditLog)
        .filter(models.AuditLog.created_at < cutoff_date)
        .delete(synchronize_session=False)
    )

    # Delete old system metrics and return number deleted
    metrics_deleted = (
        db.query(models.SystemMetrics)
        .filter(models.SystemMetrics.created_at < cutoff_date)
        .delete(synchronize_session=False)
    )

    db.commit()

    return {
        "message": "System cleanup completed",
        "audit_logs_deleted": audit_deleted,
        "metrics_deleted": metrics_deleted,
    }
