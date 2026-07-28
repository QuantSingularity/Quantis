"""
Authentication utilities re-exported from endpoints.auth
"""

from ..endpoints.auth import (
    AuditLogger,
    admin_required,
    get_current_user,
    prediction_rate_limit,
    rate_limit,
    readonly_or_above,
    require_admin,
    require_permission,
    require_verified_user,
    user_or_admin_required,
    validate_api_key,
)

__all__ = [
    "AuditLogger",
    "rate_limit",
    "get_current_user",
    "require_permission",
    "require_admin",
    "require_verified_user",
    "validate_api_key",
    "user_or_admin_required",
    "readonly_or_above",
    "admin_required",
    "prediction_rate_limit",
]
