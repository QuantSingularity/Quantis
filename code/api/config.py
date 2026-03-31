"""
Configuration management for the Quantis API - Working version
"""

from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class SecurityConfig:
    """Security configuration"""

    secret_key: str = "dev-key"
    jwt_secret: str = "dev-jwt"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    max_concurrent_sessions: int = 5


class DatabaseConfig:
    """Database configuration"""

    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600

    @staticmethod
    def get_database_url(db_type: str = "sqlite") -> str:
        return "sqlite:///./quantis.db"


class LoggingConfig:
    """Logging configuration"""

    log_level: str = "INFO"
    enable_audit_logging: bool = True


class ComplianceConfig:
    """Compliance configuration"""

    enable_data_encryption: bool = False
    enable_data_masking: bool = False
    enable_data_retention_policies: bool = False
    enable_consent_management: bool = False


class EncryptionConfig:
    """Encryption configuration"""

    data_encryption_key_name: str = "default_key"
    key_rotation_interval_days: int = 365


class Settings(BaseSettings):
    """Main application settings"""

    # App settings
    app_name: str = "Quantis API"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    # Security
    secret_key: str = Field(default="dev-key-change-in-prod")
    jwt_secret: str = Field(default="dev-jwt-change-in-prod")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Database
    database_url: str = Field(default="sqlite:///./quantis.db")

    # CORS
    cors_origins: List[str] = Field(default=["*"])
    allowed_hosts: List[str] = Field(default=["*"])

    # Redis (optional)
    redis_url: Optional[str] = None
    celery_broker_url: Optional[str] = None

    # Email (optional)
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True

    # Enable features
    enable_metrics: bool = True

    # Storage configuration
    storage_directory: str = "./models"

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
        "protected_namespaces": (),
    }


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()
        # Attach config objects as attributes (bypass Pydantic validation)
        sec = SecurityConfig()
        sec.secret_key = _settings.secret_key
        sec.jwt_secret = _settings.jwt_secret
        sec.algorithm = _settings.algorithm
        sec.access_token_expire_minutes = _settings.access_token_expire_minutes

        object.__setattr__(_settings, "security", sec)
        object.__setattr__(_settings, "database", DatabaseConfig())
        object.__setattr__(_settings, "logging", LoggingConfig())
        object.__setattr__(_settings, "compliance", ComplianceConfig())
        object.__setattr__(_settings, "encryption", EncryptionConfig())

        # For type checking, these need to exist
        _settings.security  # type: ignore
        _settings.database  # type: ignore
        _settings.logging  # type: ignore
        _settings.compliance  # type: ignore
        _settings.encryption  # type: ignore
    return _settings
