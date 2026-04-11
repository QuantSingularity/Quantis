"""
Authentication middleware with database integration
"""

import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import Depends, Header, HTTPException, Security
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from ..core.database import SessionLocal, get_db
from ..services.user_service import UserService

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")
JWT_SECRET = os.getenv(
    "JWT_SECRET", "your-secret-key-should-be-in-env-change-this-in-production"
)
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


class Roles:
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"


def create_jwt_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT token with expiration."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_jwt_token(token: str) -> Optional[dict]:
    """Decode and validate JWT token. Returns None on any error."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


async def validate_api_key(
    api_key: str = Security(API_KEY_HEADER), db: Session = Depends(get_db)
):
    """Validate API key and return user info."""
    try:
        env_api_key = os.getenv("API_SECRET")
        if env_api_key and api_key == env_api_key:
            return {"user_id": "system", "username": "system", "role": Roles.ADMIN}
        user_service = UserService(db)
        user_info = user_service.validate_api_key(api_key)
        if not user_info:
            raise HTTPException(status_code=403, detail="Invalid or expired API key")
        return user_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Authentication error: {str(e)}")


async def validate_jwt_token(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
):
    """Validate JWT token and return user info."""
    payload = decode_jwt_token(token)
    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_service = UserService(db)
    user = user_service.get_user_by_id(payload.get("user_id"))
    if not user:
        raise HTTPException(
            status_code=401,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    role_name = user.role.role_name if user.role else "user"
    return {
        "user_id": user.id,
        "username": user.username,
        "email": user.email,
        "role": role_name,
    }


async def optional_auth(
    api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    """Optional authentication — returns user info if authenticated, None otherwise."""
    if not api_key:
        return None
    try:
        return await validate_api_key(api_key, db)
    except HTTPException:
        return None


class RoleChecker:

    def __init__(self, required_roles: List[str]) -> None:
        self.required_roles = required_roles

    def __call__(self, user: dict = Depends(validate_api_key)) -> Any:
        if user.get("role") not in self.required_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required roles: {', '.join(self.required_roles)}",
            )
        return user


class RateLimiter:

    def __init__(self, requests_per_minute: int = 60) -> None:
        self.requests_per_minute = requests_per_minute
        self.request_history: Dict[Any, List[float]] = {}

    def __call__(self, user: dict = Depends(validate_api_key)) -> Any:
        user_id = user.get("user_id", "anonymous")
        if user_id not in self.request_history:
            self.request_history[user_id] = []
        current_time = time.time()
        minute_ago = current_time - 60
        self.request_history[user_id] = [
            ts for ts in self.request_history[user_id] if ts > minute_ago
        ]
        limit = (
            self.requests_per_minute * 2
            if user.get("role") == Roles.ADMIN
            else self.requests_per_minute
        )
        if len(self.request_history[user_id]) >= limit:
            raise HTTPException(
                status_code=429, detail="Rate limit exceeded. Try again later."
            )
        self.request_history[user_id].append(current_time)
        return user


class IPRateLimiter:

    def __init__(self, requests_per_minute: int = 30) -> None:
        self.requests_per_minute = requests_per_minute
        self.request_history: Dict[str, List[float]] = {}

    def __call__(self, request: Any) -> Any:
        client_ip = request.client.host
        if client_ip not in self.request_history:
            self.request_history[client_ip] = []
        current_time = time.time()
        minute_ago = current_time - 60
        self.request_history[client_ip] = [
            ts for ts in self.request_history[client_ip] if ts > minute_ago
        ]
        if len(self.request_history[client_ip]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=429, detail="Rate limit exceeded. Try again later."
            )
        self.request_history[client_ip].append(current_time)
        return True


admin_required = RoleChecker([Roles.ADMIN])
user_or_admin_required = RoleChecker([Roles.USER, Roles.ADMIN])
readonly_or_above = RoleChecker([Roles.READONLY, Roles.USER, Roles.ADMIN])
standard_rate_limit = RateLimiter(60)
prediction_rate_limit = RateLimiter(30)
public_rate_limit = IPRateLimiter(30)


class ApiKeyManager:
    """Legacy compatibility class — now uses database backend."""

    @staticmethod
    def create_api_key(
        user_id: str, role: str = Roles.USER, expiry_days: int = 30
    ) -> str:
        """Create a new API key — legacy compatibility method."""
        db = SessionLocal()
        try:
            user_service = UserService(db)
            uid = (
                int(user_id)
                if isinstance(user_id, str) and user_id.isdigit()
                else user_id
            )
            return user_service.create_api_key(uid, "Legacy Key", expiry_days)
        finally:
            db.close()

    @staticmethod
    def validate_api_key(api_key: str) -> Dict:
        """Validate API key — legacy compatibility method."""
        db = SessionLocal()
        try:
            user_service = UserService(db)
            result = user_service.validate_api_key(api_key)
            if result is None:
                raise HTTPException(
                    status_code=403, detail="Invalid or expired API key"
                )
            return result
        finally:
            db.close()

    @staticmethod
    def revoke_api_key(api_key: str) -> bool:
        """Revoke an API key — legacy compatibility method."""
        db = SessionLocal()
        try:
            user_service = UserService(db)
            return user_service.revoke_api_key(api_key)
        finally:
            db.close()

    @staticmethod
    def get_user_keys(user_id: str) -> List[str]:
        """Get all API keys for a user — legacy compatibility method."""
        db = SessionLocal()
        try:
            user_service = UserService(db)
            uid = (
                int(user_id)
                if isinstance(user_id, str) and user_id.isdigit()
                else user_id
            )
            api_keys = user_service.get_user_api_keys(uid)
            return [f"key_{key.id}" for key in api_keys]
        finally:
            db.close()
