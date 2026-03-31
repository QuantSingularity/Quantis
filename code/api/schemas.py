"""
Pydantic schemas for Quantis API with comprehensive validation
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import UUID4, BaseModel, EmailStr, Field, validator
from pydantic.types import confloat, conint, constr

from .models import DatasetStatus, ModelStatus, ModelType, NotificationType


class BaseSchema(BaseModel):
    """Base schema with common configuration"""

    class Config:
        from_attributes = True
        use_enum_values = True
        validate_assignment = True
        arbitrary_types_allowed = True


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields"""

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class UUIDMixin(BaseModel):
    """Mixin for UUID fields"""

    uuid: Optional[UUID4] = None


class UserBase(BaseSchema):
    """Base user schema"""

    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        pattern="^[a-zA-Z0-9_]+$",
        description="Username (3-50 characters, alphanumeric and underscore only)",
    )
    email: EmailStr = Field(..., description="Valid email address")
    first_name: Optional[str] = Field(None, max_length=50, description="First name")
    last_name: Optional[str] = Field(None, max_length=50, description="Last name")
    phone_number: Optional[str] = Field(
        None, pattern="^\\+?1?\\d{9,15}$", description="Phone number"
    )
    timezone: Optional[str] = Field("UTC", description="User timezone")
    role_id: Optional[int] = Field(None, description="ID of the user's role")


class UserCreate(UserBase):
    """Schema for creating a user"""

    password: constr(min_length=12, max_length=128) = Field(
        ..., description="Password (minimum 12 characters, strong complexity required)"
    )
    confirm_password: str = Field(..., description="Password confirmation")

    @validator("confirm_password")
    def passwords_match(cls: Any, v: Any, values: Any) -> Any:
        if "password" in values and v != values["password"]:
            raise ValueError("Passwords do not match")
        return v

    @validator("password")
    def validate_password_strength(cls: Any, v: Any) -> Any:
        if len(v) < 12:
            raise ValueError("Password must be at least 12 characters long")
        if not any((c.isupper() for c in v)):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any((c.islower() for c in v)):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any((c.isdigit() for c in v)):
            raise ValueError("Password must contain at least one digit")
        if not any((c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in v)):
            raise ValueError("Password must contain at least one special character")
        return v


class UserUpdate(BaseSchema):
    """Schema for updating a user"""

    first_name: Optional[str] = Field(None, max_length=50)
    last_name: Optional[str] = Field(None, max_length=50)
    phone_number: Optional[str] = Field(None, pattern="^\\+?1?\\d{9,15}$")
    timezone: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    is_mfa_enabled: Optional[bool] = Field(
        None, description="Whether MFA is enabled for the user"
    )


class UserResponse(UserBase, TimestampMixin, UUIDMixin):
    """Schema for user response"""

    id: int
    is_active: bool
    is_verified: bool
    is_mfa_enabled: bool
    last_login: Optional[datetime] = None
    full_name: Optional[str] = None
    role: Optional[str] = Field(None, description="User's role name")
    permissions: Optional[List[str]] = Field(
        None, description="List of permissions associated with the user's role"
    )

    @validator("full_name", always=True)
    def compute_full_name(cls: Any, v: Any, values: Any) -> Any:
        if values.get("first_name") and values.get("last_name"):
            return f"{values['first_name']} {values['last_name']}"
        return values.get("username")


class UserLogin(BaseSchema):
    """Schema for user login"""

    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")
    remember_me: bool = Field(False, description="Remember login session")
    mfa_code: Optional[str] = Field(
        None, description="Multi-factor authentication code"
    )


class PasswordChange(BaseSchema):
    """Schema for password change"""

    current_password: str = Field(..., description="Current password")
    new_password: constr(min_length=12, max_length=128) = Field(
        ...,
        description="New password (minimum 12 characters, strong complexity required)",
    )
    confirm_password: str = Field(..., description="New password confirmation")

    @validator("confirm_password")
    def passwords_match(cls: Any, v: Any, values: Any) -> Any:
        if "new_password" in values and v != values["new_password"]:
            raise ValueError("Passwords do not match")
        return v

    @validator("new_password")
    def validate_new_password_strength(cls: Any, v: Any) -> Any:
        if len(v) < 12:
            raise ValueError("Password must be at least 12 characters long")
        if not any((c.isupper() for c in v)):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any((c.islower() for c in v)):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any((c.isdigit() for c in v)):
            raise ValueError("Password must contain at least one digit")
        if not any((c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in v)):
            raise ValueError("Password must contain at least one special character")
        return v


class Token(BaseSchema):
    """Schema for authentication token"""

    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class TokenRefresh(BaseSchema):
    """Schema for token refresh"""

    refresh_token: str = Field(..., description="Refresh token")


class MFAEnable(BaseSchema):
    """Schema for enabling MFA"""

    otp_code: constr(min_length=6, max_length=6) = Field(
        ..., description="One-Time Password from authenticator app"
    )
    password: str = Field(..., description="User's current password for verification")


class MFADisable(BaseSchema):
    """Schema for disabling MFA"""

    otp_code: constr(min_length=6, max_length=6) = Field(
        ..., description="One-Time Password from authenticator app"
    )


class MFAResponse(BaseSchema):
    """Schema for MFA setup response"""

    qr_code_svg: str = Field(
        ..., description="Base64 encoded SVG string of the QR code"
    )
    secret: str = Field(..., description="MFA secret key (for manual entry)")
    message: str = Field(..., description="Instructions for MFA setup")


class ApiKeyBase(BaseSchema):
    """Base API key schema"""

    name: constr(min_length=1, max_length=100) = Field(..., description="API key name")
    description: Optional[str] = Field(None, description="API key description")
    expires_at: Optional[datetime] = Field(None, description="Expiration date")
    rate_limit: Optional[conint(ge=1, le=10000)] = Field(
        1000, description="Rate limit per hour"
    )
    scopes: Optional[List[str]] = Field([], description="API key scopes")
    ip_whitelist: Optional[List[str]] = Field(
        [], description="List of allowed IP addresses for this API key"
    )


class ApiKeyCreate(ApiKeyBase):
    """Schema for creating an API key"""


class ApiKeyResponse(ApiKeyBase, TimestampMixin):
    """Schema for API key response"""

    id: int
    key_preview: str = Field(..., description="First 8 characters of the key")
    is_active: bool
    last_used: Optional[datetime] = None
    usage_count: int


class ApiKeyWithSecret(ApiKeyResponse):
    """Schema for API key response with secret (only shown once)"""

    key: str = Field(..., description="Full API key (shown only once)")


class DatasetBase(BaseSchema):
    """Base dataset schema"""

    name: constr(min_length=1, max_length=100) = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    tags: Optional[List[str]] = Field([], description="Dataset tags")
    source: Optional[str] = Field(None, description="Data source")
    frequency: Optional[str] = Field(None, description="Data frequency")


class DatasetCreate(DatasetBase):
    """Schema for creating a dataset"""


class DatasetUpdate(BaseSchema):
    """Schema for updating a dataset"""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    source: Optional[str] = None
    frequency: Optional[str] = None


class DatasetResponse(DatasetBase, TimestampMixin, UUIDMixin):
    """Schema for dataset response"""

    id: int
    owner_id: int
    file_size: Optional[int] = None
    row_count: Optional[int] = None
    columns_info: Optional[Dict[str, Any]] = None
    status: DatasetStatus
    is_active: bool
    quality_score: Optional[float] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class DatasetUpload(BaseSchema):
    """Schema for dataset upload"""

    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    tags: Optional[List[str]] = Field([], description="Dataset tags")
    source: Optional[str] = Field(None, description="Data source")
    frequency: Optional[str] = Field(None, description="Data frequency")


class DatasetStats(BaseSchema):
    """Schema for dataset statistics"""

    shape: tuple = Field(..., description="Dataset shape (rows, columns)")
    columns: List[str] = Field(..., description="Column names")
    dtypes: Dict[str, str] = Field(..., description="Data types for each column")
    missing_values: Dict[str, int] = Field(
        ..., description="Missing value counts per column"
    )
    numeric_stats: Dict[str, Any] = Field(
        default={}, description="Statistical summary for numeric columns"
    )
    categorical_stats: Dict[str, Any] = Field(
        default={}, description="Statistics for categorical columns"
    )


class ModelBase(BaseSchema):
    """Base model schema"""

    name: constr(min_length=1, max_length=100) = Field(..., description="Model name")
    description: Optional[str] = Field(None, description="Model description")
    model_type: ModelType = Field(..., description="Model type")
    version: Optional[str] = Field("1.0.0", description="Model version")
    tags: Optional[List[str]] = Field([], description="Model tags")
    notes: Optional[str] = Field(None, description="Model notes")


class ModelCreate(ModelBase):
    """Schema for creating a model"""

    dataset_id: int = Field(..., description="Dataset ID")
    hyperparameters: Optional[Dict[str, Any]] = Field(
        {}, description="Model hyperparameters"
    )
    feature_columns: Optional[List[str]] = Field([], description="Feature columns")
    target_column: Optional[str] = Field(None, description="Target column")
    training_config: Optional[Dict[str, Any]] = Field(
        {}, description="Training configuration"
    )


class ModelUpdate(BaseSchema):
    """Schema for updating a model"""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    training_config: Optional[Dict[str, Any]] = None


class ModelResponse(ModelBase, TimestampMixin, UUIDMixin):
    """Schema for model response"""

    id: int
    owner_id: int
    dataset_id: int
    parent_model_id: Optional[int] = None
    file_size: Optional[int] = None
    hyperparameters: Dict[str, Any]
    feature_columns: List[str]
    target_column: Optional[str] = None
    training_config: Dict[str, Any]
    training_duration_seconds: Optional[int] = None
    training_samples: Optional[int] = None
    validation_samples: Optional[int] = None
    test_samples: Optional[int] = None
    metrics: Dict[str, Any]
    validation_score: Optional[float] = None
    test_score: Optional[float] = None
    status: ModelStatus
    is_active: bool
    is_deployed: bool
    deployment_url: Optional[str] = None
    trained_at: Optional[datetime] = None
    deployed_at: Optional[datetime] = None


class ModelTraining(BaseSchema):
    """Schema for model training request"""

    hyperparameters: Optional[Dict[str, Any]] = Field(
        {}, description="Training hyperparameters"
    )
    training_config: Optional[Dict[str, Any]] = Field(
        {}, description="Training configuration"
    )
    test_size: Optional[confloat(gt=0, lt=1)] = Field(0.2, description="Test set size")
    validation_size: Optional[confloat(gt=0, lt=1)] = Field(
        0.1, description="Validation set size"
    )
    random_state: Optional[int] = Field(
        42, description="Random state for reproducibility"
    )


class PredictionBase(BaseSchema):
    """Base prediction schema"""

    input_data: Dict[str, Any] = Field(..., description="Input data for prediction")
    tags: Optional[List[str]] = Field([], description="Prediction tags")
    notes: Optional[str] = Field(None, description="Prediction notes")


class PredictionCreate(PredictionBase):
    """Schema for creating a prediction"""

    model_id: int = Field(..., description="Model ID")


class PredictionBatch(BaseSchema):
    """Schema for batch predictions"""

    model_id: int = Field(..., description="Model ID")
    input_data: List[Dict[str, Any]] = Field(
        ..., description="List of input data for predictions"
    )
    tags: Optional[List[str]] = Field([], description="Prediction tags")


class PredictionResponse(PredictionBase, TimestampMixin, UUIDMixin):
    """Schema for prediction response"""

    id: int
    user_id: int
    model_id: int
    prediction_result: Dict[str, Any]
    confidence_score: Optional[float] = None
    prediction_interval: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    execution_time_ms: Optional[int] = None
    model_version: Optional[str] = None
    api_version: Optional[str] = None
    actual_value: Optional[float] = None
    feedback_score: Optional[float] = None
    is_validated: bool


class PredictionFeedback(BaseSchema):
    """Schema for prediction feedback"""

    actual_value: Optional[float] = Field(
        None, description="Actual value for validation"
    )
    feedback_score: Optional[confloat(ge=1, le=5)] = Field(
        None, description="Feedback score (1-5)"
    )
    notes: Optional[str] = Field(None, description="Feedback notes")


class NotificationBase(BaseSchema):
    """Base notification schema"""

    title: constr(min_length=1, max_length=200) = Field(
        ..., description="Notification title"
    )
    message: str = Field(..., description="Notification message")
    notification_type: NotificationType = Field(..., description="Notification type")
    priority: Optional[str] = Field("normal", description="Notification priority")
    category: Optional[str] = Field(None, description="Notification category")
    data: Optional[Dict[str, Any]] = Field(
        {}, description="Additional notification data"
    )


class NotificationCreate(NotificationBase):
    """Schema for creating a notification"""

    user_id: int = Field(..., description="User ID")


class NotificationResponse(NotificationBase, TimestampMixin):
    """Schema for notification response"""

    id: int
    user_id: int
    is_read: bool
    is_sent: bool
    sent_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    delivery_status: Optional[str] = None
    delivery_error: Optional[str] = None
    retry_count: int


class DataQualityReportResponse(BaseSchema, TimestampMixin):
    """Schema for data quality report response"""

    id: int
    dataset_id: int
    completeness_score: Optional[float] = None
    accuracy_score: Optional[float] = None
    consistency_score: Optional[float] = None
    validity_score: Optional[float] = None
    overall_score: Optional[float] = None
    column_analysis: Dict[str, Any]
    outliers_analysis: Dict[str, Any]
    duplicates_analysis: Dict[str, Any]
    missing_values_analysis: Dict[str, Any]
    recommendations: List[str]
    issues_found: List[str]


class MarketDataBase(BaseSchema):
    """Base market data schema"""

    symbol: constr(min_length=1, max_length=20) = Field(..., description="Stock symbol")
    exchange: Optional[str] = Field(None, description="Exchange")
    open_price: Optional[float] = Field(None, description="Opening price")
    high_price: Optional[float] = Field(None, description="High price")
    low_price: Optional[float] = Field(None, description="Low price")
    close_price: Optional[float] = Field(None, description="Closing price")
    volume: Optional[int] = Field(None, description="Trading volume")
    adjusted_close: Optional[float] = Field(None, description="Adjusted closing price")
    dividend_amount: Optional[float] = Field(None, description="Dividend amount")
    split_coefficient: Optional[float] = Field(None, description="Split coefficient")
    data_date: datetime = Field(..., description="Data date")
    source: str = Field(..., description="Data source")


class MarketDataCreate(MarketDataBase):
    """Schema for creating market data"""


class MarketDataResponse(MarketDataBase, TimestampMixin):
    """Schema for market data response"""

    id: int


class PermissionBase(BaseSchema):
    """Base permission schema"""

    permission_name: constr(min_length=1, max_length=100) = Field(
        ..., description="Name of the permission"
    )
    description: Optional[str] = Field(
        None, description="Description of the permission"
    )


class PermissionCreate(PermissionBase):
    """Schema for creating a permission"""


class PermissionResponse(PermissionBase, TimestampMixin):
    """Schema for permission response"""

    id: int


class RoleBase(BaseSchema):
    """Base role schema"""

    role_name: constr(min_length=1, max_length=50) = Field(
        ..., description="Name of the role"
    )
    description: Optional[str] = Field(None, description="Description of the role")
    permission_ids: Optional[List[int]] = Field(
        [], description="List of permission IDs associated with this role"
    )


class RoleCreate(RoleBase):
    """Schema for creating a role"""


class RoleResponse(RoleBase, TimestampMixin):
    """Schema for role response"""

    id: int
    permissions: Optional[List[PermissionResponse]] = Field(
        [], description="List of permissions associated with this role"
    )


class DataRetentionPolicyBase(BaseSchema):
    """Base data retention policy schema"""

    data_type: constr(min_length=1, max_length=100) = Field(
        ..., description="Type of data (e.g., 'audit_logs', 'predictions')"
    )
    retention_period_days: conint(ge=0) = Field(
        ..., description="Retention period in days (0 for indefinite)"
    )
    is_active: bool = Field(True, description="Whether the policy is active")
    description: Optional[str] = Field(None, description="Description of the policy")


class DataRetentionPolicyCreate(DataRetentionPolicyBase):
    """Schema for creating a data retention policy"""


class DataRetentionPolicyResponse(DataRetentionPolicyBase, TimestampMixin):
    """Schema for data retention policy response"""

    id: int


class ConsentRecordBase(BaseSchema):
    """Base consent record schema"""

    user_id: int = Field(..., description="ID of the user who gave consent")
    consent_type: constr(min_length=1, max_length=100) = Field(
        ..., description="Type of consent (e.g., 'data_processing', 'marketing_emails')"
    )
    is_active: bool = Field(True, description="Whether the consent is active")
    details: Optional[Dict[str, Any]] = Field(
        {}, description="Additional details about the consent"
    )


class ConsentRecordCreate(ConsentRecordBase):
    """Schema for creating a consent record"""


class ConsentRecordResponse(ConsentRecordBase, TimestampMixin):
    """Schema for consent record response"""

    id: int
    given_at: datetime


class DataMaskingConfigBase(BaseSchema):
    """Base data masking configuration schema"""

    field_name: constr(min_length=1, max_length=100) = Field(
        ..., description="Name of the field to mask (e.g., 'credit_card_number', 'ssn')"
    )
    masking_method: constr(min_length=1, max_length=50) = Field(
        ..., description="Masking method (e.g., 'hash', 'redact', 'partial')"
    )
    is_active: bool = Field(
        True, description="Whether the masking configuration is active"
    )
    config_details: Optional[Dict[str, Any]] = Field(
        {}, description="Method-specific configuration details"
    )


class DataMaskingConfigCreate(DataMaskingConfigBase):
    """Schema for creating a data masking configuration"""


class DataMaskingConfigResponse(DataMaskingConfigBase, TimestampMixin):
    """Schema for data masking configuration response"""

    id: int


class EncryptionKeyBase(BaseSchema):
    """Base encryption key schema"""

    key_name: constr(min_length=1, max_length=100) = Field(
        ..., description="Name of the encryption key"
    )
    key_value: constr(min_length=1, max_length=255) = Field(
        ..., description="Value of the encryption key (or identifier)"
    )
    key_type: constr(min_length=1, max_length=50) = Field(
        ..., description="Type of encryption key (e.g., 'AES', 'RSA', 'Fernet')"
    )
    is_active: bool = Field(True, description="Whether the key is active")
    expires_at: Optional[datetime] = Field(
        None, description="Expiration date of the key"
    )
    previous_key_id: Optional[int] = Field(
        None, description="ID of the previous key in rotation"
    )
    next_key_id: Optional[int] = Field(
        None, description="ID of the next key in rotation"
    )


class EncryptionKeyCreate(EncryptionKeyBase):
    """Schema for creating an encryption key"""


class EncryptionKeyResponse(EncryptionKeyBase, TimestampMixin):
    """Schema for encryption key response"""

    id: int


class TransactionBase(BaseSchema):
    """Base transaction schema"""

    amount: confloat(gt=0, le=1000000) = Field(
        ..., description="Transaction amount (must be positive and <= 1,000,000)"
    )
    transaction_type: str = Field(
        ..., description="Type of transaction (deposit, withdrawal, transfer, etc.)"
    )
    description: Optional[constr(max_length=500)] = Field(
        None, description="Transaction description"
    )
    counterparty_info: Optional[Dict[str, Any]] = Field(
        None, description="Information about the counterparty"
    )


class TransactionCreate(TransactionBase):
    """Schema for creating a transaction"""


class TransactionResponse(TransactionBase, TimestampMixin):
    """Schema for transaction response"""

    id: int
    user_id: int
    status: str = Field(..., description="Transaction status")
    risk_level: Optional[str] = Field(None, description="Risk level assessment")
    requires_approval: bool = Field(
        False, description="Whether transaction requires approval"
    )
    compliance_status: str = Field("compliant", description="Compliance status")


class FinancialSummaryResponse(BaseSchema):
    """Schema for financial summary response"""

    period: Dict[str, str] = Field(..., description="Period information")
    transaction_count: int = Field(..., description="Total number of transactions")
    total_volume: str = Field(..., description="Total transaction volume")
    by_type: Dict[str, Dict[str, Any]] = Field(
        ..., description="Breakdown by transaction type"
    )
    by_status: Dict[str, Dict[str, Any]] = Field(
        ..., description="Breakdown by transaction status"
    )
    average_amount: str = Field(..., description="Average transaction amount")
    largest_transaction: str = Field(..., description="Largest transaction amount")
    smallest_transaction: Optional[str] = Field(
        None, description="Smallest transaction amount"
    )


class InterestCalculationRequest(BaseSchema):
    """Schema for interest calculation request"""

    principal: confloat(gt=0) = Field(..., description="Principal amount")
    rate: confloat(gt=0, le=1) = Field(..., description="Interest rate (as decimal)")
    time_periods: conint(gt=0) = Field(..., description="Number of time periods")
    compound_frequency: conint(gt=0) = Field(
        1, description="Compounding frequency per period"
    )


class NPVCalculationRequest(BaseSchema):
    """Schema for NPV calculation request"""

    cash_flows: List[float] = Field(..., description="List of cash flows")
    discount_rate: confloat(gt=0, le=1) = Field(
        ..., description="Discount rate (as decimal)"
    )


class RiskAssessmentResponse(BaseSchema):
    """Schema for risk assessment response"""

    risk_level: str = Field(..., description="Risk level (low, medium, high, critical)")
    risk_score: int = Field(..., description="Numerical risk score")
    risk_factors: List[str] = Field(..., description="List of identified risk factors")
    requires_approval: bool = Field(
        ..., description="Whether transaction requires approval"
    )
    requires_additional_verification: bool = Field(
        ..., description="Whether additional verification is needed"
    )


class ComplianceLimitsResponse(BaseSchema):
    """Schema for compliance limits response"""

    daily_limits: Dict[str, str] = Field(
        ..., description="Daily transaction limits and usage"
    )
    monthly_limits: Dict[str, str] = Field(
        ..., description="Monthly transaction limits and usage"
    )
    compliance_status: str = Field(..., description="Overall compliance status")


class HealthCheck(BaseSchema):
    """Schema for health check response"""

    status: str = Field(..., description="Overall health status")
    database: str = Field(..., description="Database health status")
    redis: str = Field(..., description="Redis health status")
    timestamp: float = Field(..., description="Timestamp of health check")
