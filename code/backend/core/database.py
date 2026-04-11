"""
Database configuration and management for Quantis API
"""

import hashlib
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional

import redis.asyncio as redis
from cryptography.fernet import Fernet
from fastapi import Depends
from redis.asyncio import Redis
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from ..domain.models import (
    AuditLog,
    Base,
    ConsentRecord,
    DataMaskingConfig,
    DataRetentionPolicy,
    EncryptionKey,
)
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
database_url = settings.database_url
_is_sqlite = database_url.startswith("sqlite")
_engine_kwargs = {
    "echo": settings.logging.log_level == "DEBUG",
    "pool_pre_ping": True,
}
if not _is_sqlite:
    _engine_kwargs.update(
        {
            "poolclass": QueuePool,
            "pool_size": settings.database.pool_size,
            "max_overflow": settings.database.max_overflow,
            "pool_timeout": settings.database.pool_timeout,
            "pool_recycle": settings.database.pool_recycle,
        }
    )
engine = create_engine(database_url, **_engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
redis_client: Optional[Redis] = None
_encryption_keys: Dict[str, Fernet] = {}


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> Any:
    """Set SQLite pragmas for better performance and reliability"""
    if "sqlite" in database_url:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-64000")
        cursor.close()


@event.listens_for(Engine, "checkout")
def receive_checkout(
    dbapi_connection: Any, connection_record: Any, connection_proxy: Any
) -> Any:
    """Log database connection checkout"""
    logger.debug("Database connection checked out")


@event.listens_for(Engine, "checkin")
def receive_checkin(dbapi_connection: Any, connection_record: Any) -> Any:
    """Log database connection checkin"""
    logger.debug("Database connection checked in")


def init_db() -> Any:
    """Initialize database tables"""
    try:
        logger.info("Initializing database...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def get_db() -> Session:
    """Get database session dependency"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


async def get_redis() -> Optional[Redis]:
    """Get Redis client dependency. Returns None if Redis is not configured."""
    global redis_client
    if redis_client is None:
        if not settings.redis_url:
            return None
        try:
            redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            return None
    return redis_client


async def close_redis():
    """Close Redis connection"""
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None


class DatabaseManager:
    """Database management utilities"""

    @staticmethod
    def create_tables() -> Any:
        """Create all database tables"""
        Base.metadata.create_all(bind=engine)

    @staticmethod
    def drop_tables() -> Any:
        """Drop all database tables"""
        Base.metadata.drop_all(bind=engine)

    @staticmethod
    def get_table_info() -> Any:
        """Get information about database tables"""
        inspector = engine.inspect(engine)
        tables = {}
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            tables[table_name] = {
                "columns": [col["name"] for col in columns],
                "column_info": columns,
            }
        return tables

    @staticmethod
    def check_connection() -> Any:
        """Check database connection health"""
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False

    @staticmethod
    def get_connection_info() -> Any:
        """Get database connection information"""
        pool = engine.pool
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid(),
        }


class CacheManager:
    """Redis cache management utilities"""

    def __init__(self, redis_client: Redis) -> None:
        self.redis = redis_client

    async def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        try:
            return await self.redis.get(key)
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None

    async def set(self, key: str, value: str, expire: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            return await self.redis.set(key, value, ex=expire)
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            return bool(await self.redis.delete(key))
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            return bool(await self.redis.exists(key))
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False

    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter in cache"""
        try:
            return await self.redis.incrby(key, amount)
        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            return None

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for key"""
        try:
            return await self.redis.expire(key, seconds)
        except Exception as e:
            logger.error(f"Cache expire error for key {key}: {e}")
            return False

    async def get_keys(self, pattern: str) -> list:
        """Get keys matching pattern"""
        try:
            return await self.redis.keys(pattern)
        except Exception as e:
            logger.error(f"Cache keys error for pattern {pattern}: {e}")
            return []

    async def flush_db(self) -> bool:
        """Flush all keys from current database"""
        try:
            return await self.redis.flushdb()
        except Exception as e:
            logger.error(f"Cache flush error: {e}")
            return False


class TransactionManager:
    """Database transaction management"""

    def __init__(self, db: Session) -> None:
        self.db = db

    def __enter__(self) -> Any:
        return self.db

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        if exc_type is not None:
            self.db.rollback()
        else:
            self.db.commit()


@asynccontextmanager
async def get_cache_manager() -> AsyncGenerator[CacheManager, None]:
    """Get cache manager with automatic cleanup"""
    redis_client = await get_redis()
    cache_manager = CacheManager(redis_client)
    try:
        yield cache_manager
    finally:
        pass


def create_database_url(
    driver: str = "postgresql",
    username: str = "quantis",
    password: str = "password",
    host: str = "localhost",
    port: int = 5432,
    database: str = "quantis",
) -> str:
    """Create database URL from components"""
    return f"{driver}://{username}:{password}@{host}:{port}/{database}"


def migrate_database() -> Any:
    """Run database migrations"""
    try:
        logger.info("Running database migrations...")
        init_db()
        logger.info("Database migrations completed successfully")
    except Exception as e:
        logger.error(f"Database migration failed: {e}")
        raise


def health_check() -> dict:
    """Perform comprehensive health check"""
    health_status = {"database": False, "redis": False, "timestamp": None}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        health_status["database"] = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
    try:
        if settings.redis_url:
            import asyncio

            async def check_redis():
                rc = await get_redis()
                if rc is not None:
                    await rc.ping()
                    return True
                return False

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    health_status["redis"] = False  # skip in async context
                else:
                    health_status["redis"] = loop.run_until_complete(check_redis())
            except RuntimeError:
                health_status["redis"] = False
        else:
            health_status["redis"] = False
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
    from datetime import datetime

    health_status["timestamp"] = datetime.utcnow().isoformat()
    return health_status


class EncryptionManager:
    """Manages encryption and decryption of sensitive data."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def _get_fernet(self, key_name: str) -> Fernet:
        """Retrieves or generates a Fernet key."""
        if key_name not in _encryption_keys:
            key_record = (
                self.db.query(EncryptionKey)
                .filter_by(key_name=key_name, is_active=True)
                .first()
            )
            if not key_record:
                new_key_value = Fernet.generate_key().decode()
                key_record = EncryptionKey(
                    key_name=key_name,
                    key_value=new_key_value,
                    key_type="Fernet",
                    is_active=True,
                    expires_at=datetime.utcnow()
                    + timedelta(days=settings.encryption.key_rotation_interval_days),
                )
                self.db.add(key_record)
                self.db.commit()
                logger.info(f"Generated new encryption key: {key_name}")
            _encryption_keys[key_name] = Fernet(key_record.key_value.encode())
        return _encryption_keys[key_name]

    def encrypt(
        self, data: str, key_name: str = settings.encryption.data_encryption_key_name
    ) -> str:
        """Encrypts data using the specified key."""
        if not settings.compliance.enable_data_encryption:
            return data
        try:
            f = self._get_fernet(key_name)
            encrypted_data = f.encrypt(data.encode()).decode()
            logger.debug(f"Data encrypted with key {key_name}")
            return encrypted_data
        except Exception as e:
            logger.error(f"Encryption failed for key {key_name}: {e}")
            raise

    def decrypt(
        self,
        encrypted_data: str,
        key_name: str = settings.encryption.data_encryption_key_name,
    ) -> str:
        """Decrypts data using the specified key."""
        if not settings.compliance.enable_data_encryption:
            return encrypted_data
        try:
            f = self._get_fernet(key_name)
            decrypted_data = f.decrypt(encrypted_data.encode()).decode()
            logger.debug(f"Data decrypted with key {key_name}")
            return decrypted_data
        except Exception as e:
            logger.error(f"Decryption failed for key {key_name}: {e}")
            raise


class DataRetentionManager:
    """Manages data retention and deletion based on policies."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def apply_retention_policy(self, data_type: str, query: Any) -> Any:
        """Applies retention policy to a given query."""
        if not settings.compliance.enable_data_retention_policies:
            return query
        policy = (
            self.db.query(DataRetentionPolicy)
            .filter_by(data_type=data_type, is_active=True)
            .first()
        )
        if policy and policy.retention_period_days > 0:
            retention_date = datetime.utcnow() - timedelta(
                days=policy.retention_period_days
            )
            return query.filter(DataRetentionPolicy.created_at < retention_date)
        return query

    def delete_expired_data(self) -> Any:
        """Deletes data that has exceeded its retention period."""
        if not settings.compliance.enable_data_retention_policies:
            logger.info("Data retention policies are disabled. Skipping data deletion.")
            return
        audit_log_policy = (
            self.db.query(DataRetentionPolicy)
            .filter_by(data_type="audit_logs", is_active=True)
            .first()
        )
        if audit_log_policy and audit_log_policy.retention_period_days > 0:
            retention_date = datetime.utcnow() - timedelta(
                days=audit_log_policy.retention_period_days
            )
            deleted_count = (
                self.db.query(AuditLog)
                .filter(AuditLog.timestamp < retention_date)
                .delete()
            )
            self.db.commit()
            logger.info(f"Deleted {deleted_count} expired audit logs.")


class ConsentManager:
    """Manages user consent records."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def record_consent(
        self, user_id: int, consent_type: str, details: Optional[Dict[str, Any]] = None
    ) -> ConsentRecord:
        """Records a new consent for a user."""
        if not settings.compliance.enable_consent_management:
            logger.warning("Consent management is disabled. Consent not recorded.")
            return None
        consent = ConsentRecord(
            user_id=user_id,
            consent_type=consent_type,
            details=details or {},
            is_active=True,
        )
        self.db.add(consent)
        self.db.commit()
        self.db.refresh(consent)
        logger.info(f"Consent recorded for user {user_id}, type: {consent_type}")
        return consent

    def revoke_consent(self, user_id: int, consent_type: str) -> Any:
        """Revokes an existing consent for a user."""
        if not settings.compliance.enable_consent_management:
            logger.warning("Consent management is disabled. Consent not revoked.")
            return
        consent = (
            self.db.query(ConsentRecord)
            .filter_by(user_id=user_id, consent_type=consent_type, is_active=True)
            .first()
        )
        if consent:
            consent.is_active = False
            self.db.commit()
            logger.info(f"Consent revoked for user {user_id}, type: {consent_type}")

    def get_user_consents(self, user_id: int) -> List[ConsentRecord]:
        """Retrieves all active consents for a user."""
        return (
            self.db.query(ConsentRecord)
            .filter_by(user_id=user_id, is_active=True)
            .all()
        )


class DataMaskingManager:
    """Applies data masking based on configured policies."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.masking_configs = self._load_masking_configs()

    def _load_masking_configs(self) -> Dict[str, DataMaskingConfig]:
        """Loads active data masking configurations from the database."""
        configs = self.db.query(DataMaskingConfig).filter_by(is_active=True).all()
        return {config.field_name: config for config in configs}

    def mask_data(self, field_name: str, data: str) -> str:
        """Applies masking to a given data field based on configuration."""
        if not settings.compliance.enable_data_masking:
            return data
        config = self.masking_configs.get(field_name)
        if not config:
            return data
        method = config.masking_method
        if method == "hash":
            return hashlib.sha256(data.encode()).hexdigest()
        elif method == "redact":
            return "[REDACTED]"
        elif method == "partial":
            if len(data) > 4:
                return "*" * (len(data) - 4) + data[-4:]
            return "*" * len(data)
        else:
            logger.warning(f"Unknown masking method: {method} for field {field_name}")
            return data

    def mask_object(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Applies masking to all relevant fields in a dictionary object."""
        if not settings.compliance.enable_data_masking:
            return obj
        masked_obj = obj.copy()
        for field_name, config in self.masking_configs.items():
            if field_name in masked_obj:
                masked_obj[field_name] = self.mask_data(
                    field_name, str(masked_obj[field_name])
                )
        return masked_obj


def get_encryption_manager(db: Session = Depends(get_db)) -> EncryptionManager:
    return EncryptionManager(db)


def get_data_retention_manager(db: Session = Depends(get_db)) -> DataRetentionManager:
    return DataRetentionManager(db)


def get_consent_manager(db: Session = Depends(get_db)) -> ConsentManager:
    return ConsentManager(db)


def get_data_masking_manager(db: Session = Depends(get_db)) -> DataMaskingManager:
    return DataMaskingManager(db)
