"""
User service for user management operations
"""

from datetime import datetime, timedelta
from typing import List, Optional
from .. import models
from sqlalchemy import and_
from sqlalchemy.orm import Session


class UserService:

    def __init__(self, db: Session) -> None:
        self.db = db

    def create_user(
        self, username: str, email: str, password: str, role: str = "user"
    ) -> models.User:
        """Create a new user"""
        existing_user = (
            self.db.query(models.User)
            .filter(
                (models.models.User.username == username)
                | (models.models.User.email == email)
            )
            .first()
        )
        if existing_user:
            raise ValueError("User with this username or email already exists")
        user = models.User(
            username=username,
            email=email,
            hashed_password=models.models.User.hash_password(password),
            role=role,
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def authenticate_user(self, username: str, password: str) -> Optional[models.User]:
        """Authenticate user with username/password"""
        user = (
            self.db.query(models.User)
            .filter(
                and_(
                    models.models.User.username == username,
                    models.models.User.is_active,
                )
            )
            .first()
        )
        if user and user.verify_password(password):
            user.last_login = datetime.utcnow()
            self.db.commit()
            return user
        return None

    def get_user_by_id(self, user_id: int) -> Optional[models.User]:
        """Get user by ID"""
        return (
            self.db.query(models.User)
            .filter(
                and_(
                    models.models.User.id == user_id,
                    models.models.User.is_active,
                )
            )
            .first()
        )

    def get_user_by_username(self, username: str) -> Optional[models.User]:
        """Get user by username"""
        return (
            self.db.query(models.User)
            .filter(
                and_(
                    models.models.User.username == username,
                    models.models.User.is_active,
                )
            )
            .first()
        )

    def get_users(self, skip: int = 0, limit: int = 100) -> List[models.User]:
        """Get list of users"""
        return (
            self.db.query(models.User)
            .filter(models.models.User.is_active)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def update_user(self, user_id: int, **kwargs) -> Optional[models.User]:
        """Update user information"""
        user = self.get_user_by_id(user_id)
        if not user:
            return None
        for key, value in kwargs.items():
            if hasattr(user, key) and key != "id":
                if key == "password":
                    user.hashed_password = models.models.User.hash_password(value)
                else:
                    setattr(user, key, value)
        user.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(user)
        return user

    def deactivate_user(self, user_id: int) -> bool:
        """Deactivate user (soft delete)"""
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        user.is_active = False
        user.updated_at = datetime.utcnow()
        self.db.commit()
        return True

    def create_api_key(self, user_id: int, name: str, expires_days: int = 30) -> str:
        """Create API key for user"""
        user = self.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        key = models.ApiKey.generate_key()
        key_hash = models.ApiKey.hash_key(key)
        expires_at = (
            datetime.utcnow() + timedelta(days=expires_days)
            if expires_days > 0
            else None
        )
        api_key = models.ApiKey(
            key_hash=key_hash, user_id=user_id, name=name, expires_at=expires_at
        )
        self.db.add(api_key)
        self.db.commit()
        return key

    def validate_api_key(self, key: str) -> Optional[dict]:
        """Validate API key and return user info"""
        key_hash = models.ApiKey.hash_key(key)
        api_key = (
            self.db.query(models.ApiKey)
            .filter(and_(models.ApiKey.key_hash == key_hash, models.ApiKey.is_active))
            .first()
        )
        if not api_key:
            return None
        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            return None
        user = self.get_user_by_id(api_key.user_id)
        if not user:
            return None
        api_key.last_used = datetime.utcnow()
        self.db.commit()
        return {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "api_key_id": api_key.id,
            "expires_at": (
                api_key.expires_at.isoformat() if api_key.expires_at else None
            ),
        }

    def revoke_api_key(self, key: str) -> bool:
        """Revoke API key"""
        key_hash = models.ApiKey.hash_key(key)
        api_key = (
            self.db.query(models.ApiKey)
            .filter(models.ApiKey.key_hash == key_hash)
            .first()
        )
        if not api_key:
            return False
        api_key.is_active = False
        self.db.commit()
        return True

    def get_user_api_keys(self, user_id: int) -> List[models.ApiKey]:
        """Get all API keys for a user"""
        return (
            self.db.query(models.ApiKey)
            .filter(and_(models.ApiKey.user_id == user_id, models.ApiKey.is_active))
            .all()
        )
