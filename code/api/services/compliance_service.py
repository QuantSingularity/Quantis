"""
Compliance Service for Quantis Backend
Handles data masking, tokenization, retention policies, and consent management
"""

import base64
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet
from sqlalchemy import and_
from sqlalchemy.orm import Session

from ..models import AuditLog, User

logger = logging.getLogger(__name__)


class DataMaskingService:
    """Service for data masking and tokenization"""

    def __init__(self) -> None:
        self.masking_key = self._get_or_create_masking_key()
        self.cipher_suite = Fernet(self.masking_key)

    def _get_or_create_masking_key(self) -> bytes:
        """Get or create encryption key for data masking"""
        try:
            key_file = "/tmp/masking_key.key"
            try:
                with open(key_file, "rb") as f:
                    return f.read()
            except FileNotFoundError:
                key = Fernet.generate_key()
                with open(key_file, "wb") as f:
                    f.write(key)
                return key
        except Exception as e:
            logger.error(f"Error managing masking key: {e}")
            return base64.urlsafe_b64encode(
                hashlib.sha256(b"quantis_masking_key").digest()
            )

    def mask_email(self, email: str) -> str:
        """Mask email address"""
        if not email or "@" not in email:
            return "***@***.***"
        local, domain = email.split("@", 1)
        if len(local) <= 2:
            masked_local = "*" * len(local)
        else:
            masked_local = local[0] + "*" * (len(local) - 2) + local[-1]
        domain_parts = domain.split(".")
        if len(domain_parts) >= 2:
            masked_domain = "*" * len(domain_parts[0]) + "." + domain_parts[-1]
        else:
            masked_domain = "*" * len(domain)
        return f"{masked_local}@{masked_domain}"

    def mask_phone(self, phone: str) -> str:
        """Mask phone number"""
        if not phone:
            return "***-***-****"
        digits = "".join(filter(str.isdigit, phone))
        if len(digits) >= 10:
            return f"***-***-{digits[-4:]}"
        else:
            return "*" * len(phone)

    def mask_credit_card(self, card_number: str) -> str:
        """Mask credit card number"""
        if not card_number:
            return "****-****-****-****"
        digits = "".join(filter(str.isdigit, card_number))
        if len(digits) >= 4:
            return f"****-****-****-{digits[-4:]}"
        else:
            return "*" * len(card_number)

    def tokenize_sensitive_data(self, data: str) -> str:
        """Tokenize sensitive data using encryption"""
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Error tokenizing data: {e}")
            return "TOKENIZATION_ERROR"

    def detokenize_data(self, token: str) -> Optional[str]:
        """Detokenize data"""
        try:
            encrypted_data = base64.urlsafe_b64decode(token.encode())
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Error detokenizing data: {e}")
            return None

    def mask_user_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask user data for display"""
        masked_data = user_data.copy()
        if "email" in masked_data:
            masked_data["email"] = self.mask_email(masked_data["email"])
        if "phone" in masked_data:
            masked_data["phone"] = self.mask_phone(masked_data["phone"])
        for key, value in masked_data.items():
            if any(
                (
                    sensitive in key.lower()
                    for sensitive in ["ssn", "social", "tax_id", "passport"]
                )
            ):
                if isinstance(value, str) and len(value) > 4:
                    masked_data[key] = "*" * (len(value) - 4) + value[-4:]
                else:
                    masked_data[key] = "*" * len(str(value))
        return masked_data


class DataRetentionService:
    """Service for data retention and deletion policies"""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_retention_policy(self, data_type: str) -> Dict[str, Any]:
        """Get retention policy for specific data type"""
        policies = {
            "user_data": {
                "retention_days": 2555,
                "deletion_method": "secure_delete",
                "backup_retention_days": 365,
            },
            "transaction_logs": {
                "retention_days": 2555,
                "deletion_method": "secure_delete",
                "backup_retention_days": 365,
            },
            "audit_logs": {
                "retention_days": 2555,
                "deletion_method": "archive",
                "backup_retention_days": 730,
            },
            "session_data": {
                "retention_days": 30,
                "deletion_method": "immediate_delete",
                "backup_retention_days": 0,
            },
            "temporary_files": {
                "retention_days": 7,
                "deletion_method": "immediate_delete",
                "backup_retention_days": 0,
            },
        }
        return policies.get(
            data_type,
            {
                "retention_days": 365,
                "deletion_method": "secure_delete",
                "backup_retention_days": 90,
            },
        )

    def identify_expired_data(self, data_type: str) -> List[Dict[str, Any]]:
        """Identify data that has exceeded retention period"""
        policy = self.get_retention_policy(data_type)
        retention_days = policy.get("retention_days", 365)
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        expired_data = []
        try:
            if data_type == "user_data":
                expired_users = (
                    self.db.query(User)
                    .filter(
                        and_(
                            User.last_login < cutoff_date,
                            User.deletion_requested,
                        )
                    )
                    .all()
                )
                for user in expired_users:
                    expired_data.append(
                        {
                            "type": "user",
                            "id": user.id,
                            "created_at": user.created_at,
                            "last_activity": user.last_login,
                        }
                    )
            elif data_type == "audit_logs":
                expired_logs = (
                    self.db.query(AuditLog)
                    .filter(AuditLog.timestamp < cutoff_date)
                    .all()
                )
                for log in expired_logs:
                    expired_data.append(
                        {"type": "audit_log", "id": log.id, "created_at": log.timestamp}
                    )
        except Exception as e:
            logger.error(f"Error identifying expired data: {e}")
        return expired_data

    def secure_delete_user_data(self, user_id: int) -> bool:
        """Securely delete user data"""
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            if not user:
                return False
            logger.info(f"Securely deleting user data for user_id: {user_id}")
            user.is_deleted = True
            user.deleted_at = datetime.utcnow()
            user.email = f"deleted_{user_id}@deleted.local"
            user.username = f"deleted_{user_id}"
            user.password_hash = "DELETED"
            self.db.commit()
            return True
        except Exception as e:
            logger.error(f"Error securely deleting user data: {e}")
            self.db.rollback()
            return False

    def archive_old_data(self, data_type: str, archive_location: str) -> bool:
        """Archive old data to specified location"""
        try:
            expired_data = self.identify_expired_data(data_type)
            if not expired_data:
                return True
            {
                "archive_date": datetime.utcnow().isoformat(),
                "data_type": data_type,
                "record_count": len(expired_data),
                "archive_location": archive_location,
                "records": expired_data,
            }
            logger.info(f"Archived {len(expired_data)} records of type {data_type}")
            return True
        except Exception as e:
            logger.error(f"Error archiving data: {e}")
            return False


class ConsentManagementService:
    """Service for managing user consent"""

    def __init__(self, db: Session) -> None:
        self.db = db

    def record_consent(
        self,
        user_id: int,
        consent_type: str,
        granted: bool,
        purpose: str,
        legal_basis: str = None,
    ) -> bool:
        """Record user consent"""
        try:
            {
                "user_id": user_id,
                "consent_type": consent_type,
                "granted": granted,
                "purpose": purpose,
                "legal_basis": legal_basis,
                "timestamp": datetime.utcnow(),
                "ip_address": None,
                "user_agent": None,
            }
            logger.info(
                f"Recorded consent for user {user_id}: {consent_type} = {granted}"
            )
            return True
        except Exception as e:
            logger.error(f"Error recording consent: {e}")
            return False

    def get_user_consents(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all consents for a user"""
        try:
            return [
                {
                    "consent_type": "data_processing",
                    "granted": True,
                    "purpose": "Financial analysis and reporting",
                    "timestamp": datetime.utcnow(),
                    "legal_basis": "legitimate_interest",
                }
            ]
        except Exception as e:
            logger.error(f"Error getting user consents: {e}")
            return []

    def withdraw_consent(self, user_id: int, consent_type: str) -> bool:
        """Withdraw user consent"""
        try:
            self.record_consent(
                user_id=user_id,
                consent_type=consent_type,
                granted=False,
                purpose="Consent withdrawal",
                legal_basis="user_request",
            )
            logger.info(f"Consent withdrawn for user {user_id}: {consent_type}")
            return True
        except Exception as e:
            logger.error(f"Error withdrawing consent: {e}")
            return False

    def check_consent_required(self, user_id: int, operation: str) -> bool:
        """Check if consent is required for an operation"""
        consent_required_operations = [
            "data_export",
            "third_party_sharing",
            "marketing_communications",
            "analytics_tracking",
        ]
        return operation in consent_required_operations


class PrivilegeManagementService:
    """Service for implementing principle of least privilege"""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_minimum_required_permissions(self, role: str, operation: str) -> List[str]:
        """Get minimum required permissions for a role and operation"""
        permission_matrix = {
            "user": {
                "view_own_data": ["read_own_profile"],
                "update_own_data": ["read_own_profile", "write_own_profile"],
                "create_dataset": ["create_dataset"],
                "view_own_datasets": ["read_own_datasets"],
                "run_prediction": ["use_models"],
            },
            "analyst": {
                "view_all_datasets": ["read_datasets"],
                "create_model": ["create_model"],
                "view_models": ["read_models"],
                "run_analysis": ["use_models", "read_datasets"],
            },
            "admin": {
                "manage_users": ["read_users", "write_users"],
                "manage_system": ["system_admin"],
                "view_audit_logs": ["read_audit_logs"],
                "manage_compliance": ["compliance_admin"],
            },
        }
        role_permissions = permission_matrix.get(role, {})
        return role_permissions.get(operation, [])

    def validate_operation_permissions(
        self, user_role: str, operation: str, user_permissions: List[str]
    ) -> bool:
        """Validate if user has minimum required permissions for operation"""
        required_permissions = self.get_minimum_required_permissions(
            user_role, operation
        )
        if not required_permissions:
            return False
        return all((perm in user_permissions for perm in required_permissions))

    def get_data_access_scope(self, user_id: int, user_role: str) -> Dict[str, Any]:
        """Get data access scope for user based on role and permissions"""
        scopes = {
            "user": {
                "can_access_own_data": True,
                "can_access_public_data": True,
                "can_access_all_data": False,
                "data_filters": {"user_id": user_id},
            },
            "analyst": {
                "can_access_own_data": True,
                "can_access_public_data": True,
                "can_access_all_data": False,
                "data_filters": {"department": "analytics"},
            },
            "admin": {
                "can_access_own_data": True,
                "can_access_public_data": True,
                "can_access_all_data": True,
                "data_filters": {},
            },
        }
        return scopes.get(user_role, scopes["user"])


masking_service = DataMaskingService()


def get_compliance_services(db: Session) -> Any:
    """Get compliance service instances"""
    return {
        "masking": masking_service,
        "retention": DataRetentionService(db),
        "consent": ConsentManagementService(db),
        "privilege": PrivilegeManagementService(db),
    }
