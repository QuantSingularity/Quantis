"""
Authentication and security system for Quantis API
"""

import hashlib
import io
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional

import bcrypt as _bcrypt_lib
import pyotp
import qrcode
import redis.asyncio as redis
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.database import get_db, get_redis
from ..domain.models import ApiKey, AuditLog, User, UserSession
from ..domain.schemas import Token

logger = logging.getLogger(__name__)
settings = get_settings()
ALGORITHM = settings.security.algorithm
SECRET_KEY = settings.security.secret_key
ACCESS_TOKEN_EXPIRE_MINUTES = settings.security.access_token_expire_minutes
REFRESH_TOKEN_EXPIRE_DAYS = settings.security.refresh_token_expire_days
bearer_scheme = HTTPBearer(auto_error=False)


class SecurityManager:
    """Centralized security management"""

    def __init__(self) -> None:
        self.failed_attempts = {}

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt directly."""
        prepared = hashlib.sha256(password.encode("utf-8")).digest()
        return _bcrypt_lib.hashpw(prepared, _bcrypt_lib.gensalt()).decode("utf-8")

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its bcrypt hash."""
        prepared = hashlib.sha256(plain_password.encode("utf-8")).digest()
        return _bcrypt_lib.checkpw(prepared, hashed_password.encode("utf-8"))

    def generate_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        """Generate a JWT token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError as e:
            logger.warning(f"Token verification failed: {e}")
            return None

    def create_access_token(
        self, user_id: int, username: str, role: str, permissions: List[str]
    ) -> str:
        """Create an access token"""
        data = {
            "sub": str(user_id),
            "username": username,
            "role": role,
            "permissions": permissions,
            "type": "access",
        }
        return self.generate_token(data, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))

    def create_refresh_token(self, user_id: int, username: str) -> str:
        """Create a refresh token"""
        data = {"sub": str(user_id), "username": username, "type": "refresh"}
        return self.generate_token(data, timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))

    def generate_api_key(self) -> str:
        """Generate a new API key"""
        return f"qk_{secrets.token_urlsafe(32)}"

    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key"""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def check_rate_limit(self, identifier: str, limit: int, window: int) -> bool:
        """Check if request is within rate limit"""
        current_time = time.time()
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []
        self.failed_attempts[identifier] = [
            attempt
            for attempt in self.failed_attempts[identifier]
            if current_time - attempt < window
        ]
        if len(self.failed_attempts[identifier]) >= limit:
            return False
        self.failed_attempts[identifier].append(current_time)
        return True

    def is_account_locked(self, user: User) -> bool:
        """Check if user account is locked"""
        if user.locked_until and user.locked_until > datetime.utcnow():
            return True
        return False

    def lock_account(self, user: User, db: Session) -> Any:
        """Lock user account after failed attempts"""
        user.locked_until = datetime.utcnow() + timedelta(
            minutes=settings.security.lockout_duration_minutes
        )
        user.login_attempts = 0
        db.commit()
        logger.warning(f"Account locked for user {user.username}")

    def record_failed_login(self, user: User, db: Session) -> Any:
        """Record a failed login attempt"""
        user.login_attempts += 1
        if user.login_attempts >= settings.security.max_login_attempts:
            self.lock_account(user, db)
        else:
            db.commit()

    def record_successful_login(self, user: User, db: Session) -> Any:
        """Record a successful login"""
        user.last_login = datetime.utcnow()
        user.login_attempts = 0
        user.locked_until = None
        db.commit()

    def generate_mfa_secret(self) -> str:
        """Generates a new MFA secret."""
        return pyotp.random_base32()

    def get_mfa_uri(self, user_email: str, secret: str) -> str:
        """Generates the provisioning URI for MFA."""
        return pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email, issuer_name="Quantis"
        )

    def verify_mfa_code(self, secret: str, otp_code: str) -> bool:
        """Verifies the MFA code."""
        totp = pyotp.TOTP(secret)
        return totp.verify(otp_code)

    def generate_qr_code_svg(self, uri: str) -> str:
        """Generates a base64-encoded PNG for a QR code from a URI."""
        img = qrcode.make(uri)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        import base64

        return base64.b64encode(buffer.getvalue()).decode("utf-8")


security_manager = SecurityManager()


class RateLimiter:
    """Advanced rate limiting with Redis backend"""

    def __init__(self, redis_client: redis.Redis) -> None:
        self.redis = redis_client

    async def is_allowed(
        self, key: str, limit: int, window: int, identifier: str = "default"
    ) -> tuple:
        """
        Check if request is allowed based on rate limit.
        Returns (is_allowed, info_dict)
        """
        current_time = int(time.time())
        window_start = current_time - window

        # Use pipeline correctly: execute cleanup and count atomically
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        results = await pipe.execute()
        current_requests = results[1]

        if current_requests >= limit:
            oldest_request = await self.redis.zrange(key, 0, 0, withscores=True)
            if oldest_request:
                reset_time = int(oldest_request[0][1]) + window
                time_until_reset = max(0, reset_time - current_time)
            else:
                time_until_reset = window
            return (
                False,
                {
                    "limit": limit,
                    "remaining": 0,
                    "reset_time": time_until_reset,
                    "retry_after": time_until_reset,
                },
            )

        pipe2 = self.redis.pipeline()
        pipe2.zadd(key, {f"{current_time}:{identifier}": current_time})
        pipe2.expire(key, window)
        await pipe2.execute()

        remaining = limit - current_requests - 1
        return (
            True,
            {
                "limit": limit,
                "remaining": remaining,
                "reset_time": window,
                "retry_after": 0,
            },
        )


async def get_rate_limiter() -> RateLimiter:
    """Get rate limiter dependency"""
    redis_client = await get_redis()
    return RateLimiter(redis_client)


def rate_limit(requests: int = 100, window: int = 60) -> Any:
    """Rate limiting decorator"""

    def decorator(func):

        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if not request:
                return await func(*args, **kwargs)
            client_ip = request.client.host
            user_agent = request.headers.get("user-agent", "")
            identifier = (
                f"{client_ip}:{hashlib.md5(user_agent.encode()).hexdigest()[:8]}"
            )
            rate_limiter = await get_rate_limiter()
            is_allowed, info = await rate_limiter.is_allowed(
                f"rate_limit:{identifier}", requests, window, identifier
            )
            if not is_allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={
                        "X-RateLimit-Limit": str(info["limit"]),
                        "X-RateLimit-Remaining": str(info["remaining"]),
                        "X-RateLimit-Reset": str(info["reset_time"]),
                        "Retry-After": str(info["retry_after"]),
                    },
                )
            response = await func(*args, **kwargs)
            if hasattr(response, "headers"):
                response.headers["X-RateLimit-Limit"] = str(info["limit"])
                response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
                response.headers["X-RateLimit-Reset"] = str(info["reset_time"])
            return response

        return wrapper

    return decorator


async def get_current_user_from_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
    mfa_code: Optional[str] = None,
) -> User:
    """Get current user from JWT token"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = security_manager.verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = (
        db.query(User)
        .filter(
            User.id == int(user_id), User.is_active == True, User.is_deleted == False
        )
        .first()
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if security_manager.is_account_locked(user):
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED, detail="Account is temporarily locked"
        )
    if user.is_mfa_enabled:
        if not mfa_code:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="MFA code required"
            )
        # Bug fix: was `user.mfa_secret == False` which compares object to bool
        if not user.mfa_secret:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="MFA enabled but secret not found",
            )
        if not security_manager.verify_mfa_code(user.mfa_secret, mfa_code):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Invalid MFA code"
            )
    return user


async def get_current_user_from_api_key(
    request: Request, db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user from API key"""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return None
    key_hash = security_manager.hash_api_key(api_key)
    api_key_obj = (
        db.query(ApiKey)
        .filter(
            ApiKey.key_hash == key_hash,
            ApiKey.is_active == True,
            ApiKey.is_deleted == False,
        )
        .first()
    )
    if not api_key_obj:
        return None
    if api_key_obj.is_expired():
        return None
    client_ip = request.client.host
    if api_key_obj.ip_whitelist and client_ip not in api_key_obj.ip_whitelist:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API Key not authorized from this IP address",
        )
    user = (
        db.query(User)
        .filter(
            User.id == api_key_obj.user_id,
            User.is_active == True,
            User.is_deleted == False,
        )
        .first()
    )
    if not user:
        return None
    api_key_obj.last_used = datetime.utcnow()
    api_key_obj.usage_count += 1
    db.commit()
    return user


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Get current user from either JWT token or API key"""
    user = await get_current_user_from_api_key(request, db)
    if user:
        return user
    return await get_current_user_from_token(credentials, db)


def require_permission(required_permissions: Any) -> Any:
    """
    Decorator to require specific user permission(s).

    Accepts either a single permission name (``"read_datasets"``) or a list
    of permission names (``["read_datasets", "read_dataset"]``) — every
    call site in this codebase uses the single-string form, so a bare
    string is normalized into a one-element list here rather than being
    iterated character-by-character.
    """
    if isinstance(required_permissions, str):
        required_permissions = [required_permissions]

    def decorator(func):

        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = None
            # Check positional args
            for arg in args:
                if isinstance(arg, User):
                    current_user = arg
                    break
            # Check keyword args (FastAPI passes dependencies as kwargs)
            if not current_user:
                for v in kwargs.values():
                    if isinstance(v, User):
                        current_user = v
                        break
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )
            user_permissions = (
                set([p.permission_name for p in current_user.role.permissions])
                if current_user.role
                else set()
            )
            if not all((perm in user_permissions for perm in required_permissions)):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions",
                )
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_admin(current_user: User = Depends(get_current_user)) -> Any:
    """Dependency to require admin role"""
    role_name = current_user.role.role_name if current_user.role else ""
    if role_name != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
        )
    return current_user


def require_verified_user(current_user: User = Depends(get_current_user)) -> Any:
    """Dependency to require verified user"""
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Email verification required"
        )
    return current_user


class AuditLogger:
    """Audit logging for security events"""

    @staticmethod
    def log_security_event(
        db: Session,
        user_id: Optional[int],
        action: str,
        resource_type: str = "security",
        resource_id: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
        request: Optional[Request] = None,
    ) -> Any:
        """
        Log a security-relevant event (e.g. blocked transactions, suspicious
        activity). Thin wrapper around ``log_event`` with a security-oriented
        default resource type.
        """
        return AuditLogger.log_event(
            db=db,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id is not None else None,
            details=details,
            request=request,
        )

    @staticmethod
    def log_event(
        db: Session,
        user_id: Optional[int],
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        resource_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request: Optional[Request] = None,
        status_code: Optional[int] = None,
    ) -> Any:
        """Log an audit event"""
        audit_log = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            details=details or {},
            status_code=status_code,
        )
        if request:
            audit_log.ip_address = request.client.host
            audit_log.user_agent = request.headers.get("user-agent")
            audit_log.endpoint = str(request.url.path)
            audit_log.method = request.method
        db.add(audit_log)
        db.commit()

    @staticmethod
    def log_login_attempt(
        db: Session,
        username: str,
        success: bool,
        request: Request,
        user_id: Optional[int] = None,
        failure_reason: Optional[str] = None,
    ) -> Any:
        """Log a login attempt"""
        AuditLogger.log_event(
            db=db,
            user_id=user_id,
            action="user_login",
            resource_type="authentication",
            resource_name=username,
            request=request,
            status_code=200 if success else 401,
            details={"success": success, "failure_reason": failure_reason},
        )

    @staticmethod
    def log_mfa_event(
        db: Session,
        user_id: int,
        action: str,
        request: Request,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Log an MFA related event"""
        AuditLogger.log_event(
            db=db,
            user_id=user_id,
            action=action,
            resource_type="mfa",
            resource_id=str(user_id),
            resource_name=f"MFA for user {user_id}",
            request=request,
            status_code=200 if success else 400,
            details=details,
        )


async def get_current_user_with_mfa(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Dependency to get current user, requiring MFA if enabled"""
    mfa_code = request.headers.get("X-MFA-Code")
    return await get_current_user_from_token(credentials, db, mfa_code)


async def get_current_user_for_mfa_setup(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Dependency to get current user without MFA check for MFA setup/disable endpoints"""
    return await get_current_user_from_token(credentials, db, mfa_code=None)


async def create_user_session(
    db: Session,
    user: User,
    access_token: str,
    refresh_token: str,
    request: Request,
    max_concurrent_sessions: int = settings.security.max_concurrent_sessions,
) -> UserSession:
    """Create a new user session, handling concurrent sessions"""
    active_sessions = (
        db.query(UserSession)
        .filter(UserSession.user_id == user.id, UserSession.is_active == True)
        .order_by(UserSession.last_activity.asc())
        .all()
    )
    if len(active_sessions) >= max_concurrent_sessions:
        for i in range(len(active_sessions) - max_concurrent_sessions + 1):
            active_sessions[i].is_active = False
            AuditLogger.log_event(
                db=db,
                user_id=user.id,
                action="session_invalidation",
                resource_type="user_session",
                resource_id=str(active_sessions[i].id),
                resource_name=f"Session {active_sessions[i].id} for {user.username}",
                request=request,
                details={"reason": "Max concurrent sessions exceeded"},
            )
    existing_session_same_client = (
        db.query(UserSession)
        .filter(
            UserSession.user_id == user.id,
            UserSession.ip_address == request.client.host,
            UserSession.user_agent == request.headers.get("user-agent"),
            UserSession.is_active == True,
        )
        .first()
    )
    if existing_session_same_client:
        existing_session_same_client.is_active = False
        AuditLogger.log_event(
            db=db,
            user_id=user.id,
            action="session_invalidation",
            resource_type="user_session",
            resource_id=str(existing_session_same_client.id),
            resource_name=f"Session {existing_session_same_client.id} for {user.username}",
            request=request,
            details={"reason": "New session from same client fingerprint"},
        )
    session = UserSession(
        user_id=user.id,
        session_token=access_token,
        refresh_token=refresh_token,
        expires_at=datetime.utcnow()
        + timedelta(days=settings.security.refresh_token_expire_days),
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent"),
        is_active=True,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate user by username/email and password"""
    user = (
        db.query(User)
        .filter((User.username == username) | (User.email == username))
        .first()
    )
    if not user:
        return None
    if security_manager.is_account_locked(user):
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail="Account is temporarily locked due to too many failed login attempts.",
        )
    if not security_manager.verify_password(password, user.hashed_password):
        security_manager.record_failed_login(user, db)
        return None
    security_manager.record_successful_login(user, db)
    return user


def create_tokens(user: User) -> Token:
    """Create access and refresh tokens for a user"""
    user_permissions = []
    if user.role and user.role.permissions:
        user_permissions = [p.permission_name for p in user.role.permissions]
    role_name = user.role.role_name if user.role else "user"
    access_token = security_manager.create_access_token(
        user_id=user.id,
        username=user.username,
        role=role_name,
        permissions=user_permissions,
    )
    refresh_token = security_manager.create_refresh_token(
        user_id=user.id, username=user.username
    )
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.security.access_token_expire_minutes * 60,
    )


def refresh_access_token(db: Session, refresh_token: str) -> Optional[Token]:
    """Refresh access token using a valid refresh token"""
    payload = security_manager.verify_token(refresh_token)
    if not payload or payload.get("type") != "refresh":
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    user = (
        db.query(User)
        .filter(
            User.id == int(user_id), User.is_active == True, User.is_deleted == False
        )
        .first()
    )
    if not user:
        return None
    session = (
        db.query(UserSession)
        .filter(
            UserSession.user_id == user.id,
            UserSession.refresh_token == refresh_token,
            UserSession.is_active == True,
        )
        .first()
    )
    if not session or session.is_expired():
        if session:
            session.is_active = False
            db.commit()
        return None
    session.last_activity = datetime.utcnow()
    db.commit()
    return create_tokens(user)


# Router for authentication endpoints
from fastapi import APIRouter
from fastapi import status as _status

from ..domain.schemas import (
    ApiKeyCreate,
    ApiKeyResponse,
    ApiKeyWithSecret,
    MFADisable,
    MFAEnable,
    MFAResponse,
    PasswordChange,
    PasswordResetConfirm,
    PasswordResetRequest,
    Token,
    TokenRefresh,
    UserCreate,
    UserLogin,
    UserResponse,
)
from ..services.user_service import UserService

router = APIRouter()


@router.post(
    "/register", response_model=UserResponse, status_code=_status.HTTP_201_CREATED
)
async def register(
    payload: UserCreate, request: Request, db: Session = Depends(get_db)
) -> Any:
    """Register a new user account."""
    user_service = UserService(db)
    try:
        user = user_service.create_user(
            username=payload.username,
            email=payload.email,
            password=payload.password,
        )
    except ValueError as e:
        raise HTTPException(status_code=_status.HTTP_409_CONFLICT, detail=str(e))
    if payload.first_name or payload.last_name or payload.phone_number:
        user_service.update_user(
            user.id,
            first_name=payload.first_name,
            last_name=payload.last_name,
            phone_number=payload.phone_number,
        )
        db.refresh(user)
    AuditLogger.log_event(
        db=db,
        user_id=user.id,
        action="user_register",
        resource_type="user",
        resource_id=str(user.id),
        resource_name=user.username,
        request=request,
        status_code=201,
    )
    return user


@router.post("/login", response_model=Token)
async def login(
    payload: UserLogin, request: Request, db: Session = Depends(get_db)
) -> Any:
    """Authenticate a user and return access/refresh tokens."""
    try:
        user = authenticate_user(db, payload.username, payload.password)
    except HTTPException:
        AuditLogger.log_login_attempt(
            db=db,
            username=payload.username,
            success=False,
            request=request,
            failure_reason="account_locked",
        )
        raise
    if not user:
        AuditLogger.log_login_attempt(
            db=db,
            username=payload.username,
            success=False,
            request=request,
            failure_reason="invalid_credentials",
        )
        raise HTTPException(
            status_code=_status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    if user.is_mfa_enabled:
        if not payload.mfa_code:
            raise HTTPException(
                status_code=_status.HTTP_403_FORBIDDEN,
                detail="MFA code required",
            )
        if not user.mfa_secret or not security_manager.verify_mfa_code(
            user.mfa_secret, payload.mfa_code
        ):
            AuditLogger.log_login_attempt(
                db=db,
                username=payload.username,
                success=False,
                request=request,
                user_id=user.id,
                failure_reason="invalid_mfa_code",
            )
            raise HTTPException(
                status_code=_status.HTTP_403_FORBIDDEN, detail="Invalid MFA code"
            )
    tokens = create_tokens(user)
    await create_user_session(
        db=db,
        user=user,
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        request=request,
    )
    AuditLogger.log_login_attempt(
        db=db, username=payload.username, success=True, request=request, user_id=user.id
    )
    return tokens


@router.post("/refresh", response_model=Token)
async def refresh(payload: TokenRefresh, db: Session = Depends(get_db)) -> Any:
    """Exchange a valid refresh token for a new access/refresh token pair."""
    tokens = refresh_access_token(db, payload.refresh_token)
    if not tokens:
        raise HTTPException(
            status_code=_status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    return tokens


@router.post("/logout", status_code=_status.HTTP_204_NO_CONTENT, response_model=None)
async def logout(
    payload: TokenRefresh,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Invalidate the session associated with the given refresh token."""
    session = (
        db.query(UserSession)
        .filter(
            UserSession.user_id == current_user.id,
            UserSession.refresh_token == payload.refresh_token,
        )
        .first()
    )
    if session:
        session.is_active = False
        db.commit()
    AuditLogger.log_event(
        db=db,
        user_id=current_user.id,
        action="user_logout",
        resource_type="user_session",
        resource_name=current_user.username,
        request=request,
    )
    return None


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)) -> Any:
    """Return the currently authenticated user's profile."""
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_me(
    payload: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Update the currently authenticated user's profile."""
    user_service = UserService(db)
    allowed_fields = {
        "first_name",
        "last_name",
        "phone_number",
        "timezone",
        "preferences",
    }
    updates = {k: v for k, v in payload.items() if k in allowed_fields}
    updated_user = user_service.update_user(current_user.id, **updates)
    if not updated_user:
        raise HTTPException(
            status_code=_status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    return updated_user


@router.post(
    "/change-password", status_code=_status.HTTP_204_NO_CONTENT, response_model=None
)
async def change_password(
    payload: PasswordChange,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Change the currently authenticated user's password."""
    if not security_manager.verify_password(
        payload.current_password, current_user.hashed_password
    ):
        raise HTTPException(
            status_code=_status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )
    current_user.hashed_password = security_manager.hash_password(payload.new_password)
    db.commit()
    AuditLogger.log_event(
        db=db,
        user_id=current_user.id,
        action="password_change",
        resource_type="user",
        resource_id=str(current_user.id),
        resource_name=current_user.username,
        request=request,
    )
    return None


@router.post("/mfa/setup", response_model=MFAResponse)
async def setup_mfa(
    request: Request,
    current_user: User = Depends(get_current_user_for_mfa_setup),
    db: Session = Depends(get_db),
) -> Any:
    """Generate a new MFA secret and QR code for the current user (not yet enabled)."""
    if current_user.is_mfa_enabled:
        raise HTTPException(
            status_code=_status.HTTP_400_BAD_REQUEST, detail="MFA is already enabled"
        )
    secret = security_manager.generate_mfa_secret()
    current_user.mfa_secret = secret
    db.commit()
    uri = security_manager.get_mfa_uri(current_user.email, secret)
    qr_code_svg = security_manager.generate_qr_code_svg(uri)
    AuditLogger.log_mfa_event(
        db=db,
        user_id=current_user.id,
        action="mfa_setup_initiated",
        request=request,
        success=True,
    )
    return MFAResponse(
        qr_code_svg=qr_code_svg,
        secret=secret,
        message="Scan the QR code with your authenticator app, then confirm with a one-time code to enable MFA.",
    )


@router.post("/mfa/enable", response_model=UserResponse)
async def enable_mfa(
    payload: MFAEnable,
    request: Request,
    current_user: User = Depends(get_current_user_for_mfa_setup),
    db: Session = Depends(get_db),
) -> Any:
    """Confirm MFA setup with a one-time code and enable it for the account."""
    if not security_manager.verify_password(
        payload.password, current_user.hashed_password
    ):
        raise HTTPException(
            status_code=_status.HTTP_400_BAD_REQUEST, detail="Password is incorrect"
        )
    if not current_user.mfa_secret:
        raise HTTPException(
            status_code=_status.HTTP_400_BAD_REQUEST,
            detail="MFA setup has not been initiated. Call /auth/mfa/setup first.",
        )
    if not security_manager.verify_mfa_code(current_user.mfa_secret, payload.otp_code):
        AuditLogger.log_mfa_event(
            db=db,
            user_id=current_user.id,
            action="mfa_enable_failed",
            request=request,
            success=False,
        )
        raise HTTPException(
            status_code=_status.HTTP_400_BAD_REQUEST, detail="Invalid MFA code"
        )
    current_user.is_mfa_enabled = True
    db.commit()
    db.refresh(current_user)
    AuditLogger.log_mfa_event(
        db=db,
        user_id=current_user.id,
        action="mfa_enabled",
        request=request,
        success=True,
    )
    return current_user


@router.post("/mfa/disable", response_model=UserResponse)
async def disable_mfa(
    payload: MFADisable,
    request: Request,
    current_user: User = Depends(get_current_user_for_mfa_setup),
    db: Session = Depends(get_db),
) -> Any:
    """Disable MFA for the current user after verifying a one-time code."""
    if not current_user.is_mfa_enabled or not current_user.mfa_secret:
        raise HTTPException(
            status_code=_status.HTTP_400_BAD_REQUEST, detail="MFA is not enabled"
        )
    if not security_manager.verify_mfa_code(current_user.mfa_secret, payload.otp_code):
        raise HTTPException(
            status_code=_status.HTTP_400_BAD_REQUEST, detail="Invalid MFA code"
        )
    current_user.is_mfa_enabled = False
    current_user.mfa_secret = None
    db.commit()
    db.refresh(current_user)
    AuditLogger.log_mfa_event(
        db=db,
        user_id=current_user.id,
        action="mfa_disabled",
        request=request,
        success=True,
    )
    return current_user


@router.post(
    "/forgot-password", status_code=_status.HTTP_202_ACCEPTED, response_model=None
)
async def forgot_password(
    payload: PasswordResetRequest, request: Request, db: Session = Depends(get_db)
) -> Any:
    """
    Request a password reset email. Always returns 202 regardless of whether
    the email exists, to avoid leaking account existence. In debug mode the
    reset token is also returned in the response body so the flow can be
    exercised without a configured SMTP server.
    """
    user = (
        db.query(User)
        .filter(User.email == payload.email, User.is_active == True)
        .first()
    )
    response_body: Dict[str, Any] = {
        "message": "If an account with that email exists, a reset link has been sent."
    }
    if not user:
        return response_body

    reset_token = security_manager.generate_token(
        {"sub": str(user.id), "type": "password_reset"}, timedelta(minutes=30)
    )
    AuditLogger.log_event(
        db=db,
        user_id=user.id,
        action="password_reset_requested",
        resource_type="user",
        resource_id=str(user.id),
        resource_name=user.username,
        request=request,
    )

    try:
        from ..services.notification_service import NotificationService

        notification_service = NotificationService(db)
        if notification_service._is_email_configured():
            notification_service.send_email_notification(
                to_email=user.email,
                subject="Reset your Quantis password",
                message=(
                    f"Hi {user.username},\n\n"
                    f"Use this token to reset your password (valid for 30 minutes):\n\n"
                    f"{reset_token}\n\n"
                    "If you did not request this, you can safely ignore this email."
                ),
            )
    except Exception as e:
        logger.warning(f"Failed to send password reset email: {e}")

    if settings.debug:
        response_body["reset_token"] = reset_token
    return response_body


@router.post(
    "/reset-password", status_code=_status.HTTP_204_NO_CONTENT, response_model=None
)
async def reset_password(
    payload: PasswordResetConfirm, request: Request, db: Session = Depends(get_db)
) -> Any:
    """Reset a user's password using a valid password-reset token."""
    token_data = security_manager.verify_token(payload.token)
    if not token_data or token_data.get("type") != "password_reset":
        raise HTTPException(
            status_code=_status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )
    user_id = token_data.get("sub")
    user = (
        db.query(User).filter(User.id == int(user_id), User.is_active == True).first()
    )
    if not user:
        raise HTTPException(
            status_code=_status.HTTP_400_BAD_REQUEST, detail="Invalid reset token"
        )

    user.hashed_password = security_manager.hash_password(payload.new_password)
    user.login_attempts = 0
    user.locked_until = None
    db.commit()

    # Invalidate all active sessions for this user as a security precaution.
    db.query(UserSession).filter(
        UserSession.user_id == user.id, UserSession.is_active == True
    ).update({"is_active": False})
    db.commit()

    AuditLogger.log_event(
        db=db,
        user_id=user.id,
        action="password_reset_completed",
        resource_type="user",
        resource_id=str(user.id),
        resource_name=user.username,
        request=request,
    )
    return None


async def validate_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    JWT-compatible drop-in replacement for the legacy API-key-only
    ``middleware.auth.validate_api_key`` dependency.

    Preserves that dependency's shared-secret system bypass (an
    ``X-API-Key`` header matching the ``API_SECRET`` environment variable
    authenticates as a system/admin identity — used by trusted internal
    services and CI), and otherwise defers to ``get_current_user``, which
    accepts either a JWT bearer token or a database-registered API key.
    """
    api_key_header = request.headers.get("X-API-Key")
    env_api_key = os.getenv("API_SECRET")
    if env_api_key and api_key_header and api_key_header == env_api_key:
        return {
            "user_id": "system",
            "username": "system",
            "email": "system@quantis.local",
            "role": "admin",
        }

    current_user = await get_current_user(request, credentials, db)
    role_name = current_user.role.role_name if current_user.role else "user"
    return {
        "user_id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "role": role_name,
    }


def _require_role_dict(allowed_roles: List[str]) -> Any:
    async def dependency(
        current_user: Dict[str, Any] = Depends(validate_api_key),
    ) -> Dict[str, Any]:
        if current_user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=_status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {', '.join(allowed_roles)}",
            )
        return current_user

    return dependency


# JWT-compatible role-gated dependencies, mirroring middleware.auth's
# RoleChecker-based `user_or_admin_required` / `readonly_or_above` / `admin_required`.
user_or_admin_required = _require_role_dict(["user", "admin"])
readonly_or_above = _require_role_dict(["readonly", "user", "admin"])
admin_required = _require_role_dict(["admin"])


async def prediction_rate_limit(
    current_user: Dict[str, Any] = Depends(validate_api_key),
) -> Dict[str, Any]:
    """JWT-compatible rate limiter for prediction endpoints (30 req/min per identity)."""
    identifier = f"prediction:{current_user['user_id']}"
    if not security_manager.check_rate_limit(identifier, limit=30, window=60):
        raise HTTPException(
            status_code=_status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later.",
        )
    return current_user


@router.get("/api-keys", response_model=List[ApiKeyResponse])
async def list_api_keys(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> Any:
    """List all active API keys for the current user."""
    user_service = UserService(db)
    keys = user_service.get_user_api_keys(current_user.id)
    return [
        ApiKeyResponse(
            id=k.id,
            name=k.name,
            description=getattr(k, "description", None),
            expires_at=k.expires_at,
            rate_limit=getattr(k, "rate_limit", 1000),
            scopes=getattr(k, "scopes", []) or [],
            ip_whitelist=getattr(k, "ip_whitelist", []) or [],
            key_preview=f"{k.key_hash[:8]}...",
            is_active=k.is_active,
            last_used=k.last_used,
            usage_count=getattr(k, "usage_count", 0) or 0,
            created_at=k.created_at,
            updated_at=k.updated_at,
        )
        for k in keys
    ]


@router.post(
    "/api-keys", response_model=ApiKeyWithSecret, status_code=_status.HTTP_201_CREATED
)
async def create_api_key(
    payload: ApiKeyCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Create a new API key for the current user. The full key is only ever shown once."""
    user_service = UserService(db)
    expires_days = 0
    if payload.expires_at:
        delta = payload.expires_at - datetime.utcnow()
        expires_days = max(delta.days, 1)
    raw_key = user_service.create_api_key(
        current_user.id, payload.name, expires_days=expires_days or 30
    )
    created = (
        db.query(ApiKey)
        .filter(ApiKey.user_id == current_user.id)
        .order_by(ApiKey.id.desc())
        .first()
    )
    if not created:
        raise HTTPException(
            status_code=_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve created API key",
        )
    AuditLogger.log_event(
        db=db,
        user_id=current_user.id,
        action="api_key_created",
        resource_type="api_key",
        resource_id=str(created.id),
        resource_name=payload.name,
        request=request,
    )
    return ApiKeyWithSecret(
        id=created.id,
        name=created.name,
        description=payload.description,
        expires_at=created.expires_at,
        rate_limit=payload.rate_limit,
        scopes=payload.scopes or [],
        ip_whitelist=payload.ip_whitelist or [],
        key_preview=f"{raw_key[:8]}...",
        is_active=created.is_active,
        last_used=created.last_used,
        usage_count=0,
        created_at=created.created_at,
        updated_at=created.updated_at,
        key=raw_key,
    )


@router.delete(
    "/api-keys/{key_id}", status_code=_status.HTTP_204_NO_CONTENT, response_model=None
)
async def revoke_api_key(
    key_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Revoke one of the current user's API keys."""
    key = (
        db.query(ApiKey)
        .filter(ApiKey.id == key_id, ApiKey.user_id == current_user.id)
        .first()
    )
    if not key:
        raise HTTPException(
            status_code=_status.HTTP_404_NOT_FOUND, detail="API key not found"
        )
    key.is_active = False
    db.commit()
    AuditLogger.log_event(
        db=db,
        user_id=current_user.id,
        action="api_key_revoked",
        resource_type="api_key",
        resource_id=str(key_id),
        resource_name=key.name,
        request=request,
    )
    return None
