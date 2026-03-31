"""
Dataset management endpoints for Quantis API
"""

import hashlib
import logging
import os
import tempfile
from typing import List

import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from ..auth import AuditLogger, get_current_user, require_permission
from ..config import Settings, get_settings
from ..database import (
    DataMaskingManager,
    DataRetentionManager,
    EncryptionManager,
    get_data_masking_manager,
    get_data_retention_manager,
    get_db,
    get_encryption_manager,
)
from ..models import Dataset, DatasetStatus, User
from ..schemas import (
    DatasetCreate,
    DatasetResponse,
    DatasetStats,
    DatasetUpdate,
    DatasetUpload,
)
from ..services.dataset_service import DatasetService

logger = logging.getLogger(__name__)
settings: Settings = get_settings()

router = APIRouter()


# ------------------------
# Dataset CRUD endpoints
# ------------------------
@router.post("/", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
@require_permission("create_dataset")
async def create_dataset(
    dataset_data: DatasetCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new dataset record (metadata only)."""
    dataset_service = DatasetService(db)
    try:
        dataset = dataset_service.create_dataset_record(
            name=dataset_data.name,
            description=dataset_data.description,
            owner_id=current_user.id,
            tags=dataset_data.tags,
            source=dataset_data.source,
            frequency=dataset_data.frequency,
        )
        AuditLogger.log_event(
            db=db,
            user_id=current_user.id,
            action="create_dataset_record",
            resource_type="dataset",
            resource_id=str(dataset.id),
            resource_name=dataset.name,
            request=request,
        )
        logger.info(
            f"Dataset record created: {dataset.name} by user {current_user.username}"
        )
        return DatasetResponse.from_orm(dataset)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create dataset record: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create dataset record",
        )


@router.post("/upload", response_model=DatasetResponse)
@require_permission("upload_dataset")
async def upload_dataset(
    request: Request,
    dataset_upload: DatasetUpload = Depends(),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    encryption_manager: EncryptionManager = Depends(get_encryption_manager),
):
    """Upload a dataset file and create/update its record."""
    dataset_service = DatasetService(db)

    # Validate file type and size
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in settings.allowed_file_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(settings.allowed_file_types)}",
        )

    file_content = await file.read()
    if len(file_content) > settings.max_upload_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds limit of {settings.max_upload_size / (1024 * 1024):.0f}MB",
        )

    os.makedirs(settings.upload_directory, exist_ok=True)
    unique_filename = (
        f"{current_user.id}_{dataset_upload.name}_{os.urandom(8).hex()}{file_extension}"
    )
    file_path = os.path.join(settings.upload_directory, unique_filename)

    try:
        # Encrypt file if enabled
        if settings.compliance.enable_data_encryption:
            encrypted_content = encryption_manager.encrypt(
                file_content.decode("utf-8", errors="ignore")
            )
            with open(file_path, "w", encoding="utf-8") as buffer:
                buffer.write(encrypted_content)
        else:
            with open(file_path, "wb") as buffer:
                buffer.write(file_content)

        file_hash = hashlib.sha256(file_content).hexdigest()
        file_size = len(file_content)

        # Extract dataset metadata
        df = None
        try:
            if file_extension == ".csv":
                df = pd.read_csv(file_path)
            elif file_extension == ".json":
                df = pd.read_json(file_path)
            elif file_extension in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
            columns_info = (
                {col: str(df[col].dtype) for col in df.columns}
                if df is not None
                else {}
            )
            row_count = len(df) if df is not None else 0
        except Exception as e:
            logger.warning(f"Could not parse dataset file for metadata: {e}")
            columns_info = {}
            row_count = 0

        dataset = dataset_service.create_dataset_record(
            name=dataset_upload.name,
            description=dataset_upload.description,
            owner_id=current_user.id,
            file_path=file_path,
            file_size=file_size,
            file_hash=file_hash,
            columns_info=columns_info,
            row_count=row_count,
            status=DatasetStatus.READY,
            tags=dataset_upload.tags,
            source=dataset_upload.source,
            frequency=dataset_upload.frequency,
        )

        AuditLogger.log_event(
            db=db,
            user_id=current_user.id,
            action="upload_dataset",
            resource_type="dataset",
            resource_id=str(dataset.id),
            resource_name=dataset.name,
            request=request,
        )
        logger.info(
            f"Dataset uploaded and processed: {dataset.name} by user {current_user.username}"
        )
        return DatasetResponse.from_orm(dataset)

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"Failed to upload dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload dataset: {str(e)}",
        )


# ------------------------
# Dataset Read Endpoints
# ------------------------
@router.get("/", response_model=List[DatasetResponse])
@require_permission("read_datasets")
async def get_datasets(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    request: Request = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    data_retention_manager: DataRetentionManager = Depends(get_data_retention_manager),
):
    """Get datasets for the current user or all datasets for admin."""
    DatasetService(db)

    if require_permission("read_all_datasets")(current_user):
        query = db.query(Dataset).filter(not Dataset.is_deleted)
    else:
        query = db.query(Dataset).filter(
            Dataset.owner_id == current_user.id, not Dataset.is_deleted
        )

    query = data_retention_manager.apply_retention_policy("datasets", query)
    datasets = query.offset(skip).limit(limit).all()

    AuditLogger.log_event(
        db=db,
        user_id=current_user.id,
        action="read_datasets",
        resource_type="dataset",
        resource_name=(
            "all_datasets"
            if require_permission("read_all_datasets")(current_user)
            else "user_datasets"
        ),
        request=request,
    )
    return [DatasetResponse.from_orm(dataset) for dataset in datasets]


@router.get("/{dataset_id}", response_model=DatasetResponse)
@require_permission("read_dataset")
async def get_dataset(
    dataset_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    data_masking_manager: DataMaskingManager = Depends(get_data_masking_manager),
):
    """Get dataset by ID."""
    dataset_service = DatasetService(db)
    dataset = dataset_service.get_dataset_by_id(dataset_id)

    if not dataset or dataset.is_deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found"
        )

    if not (
        dataset.owner_id == current_user.id
        or require_permission("read_all_datasets")(current_user)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    dataset_response = DatasetResponse.from_orm(dataset)

    AuditLogger.log_event(
        db=db,
        user_id=current_user.id,
        action="read_dataset_by_id",
        resource_type="dataset",
        resource_id=str(dataset_id),
        resource_name=dataset.name,
        request=request,
    )
    return dataset_response


# ------------------------
# Dataset Update/Delete
# ------------------------
@router.put("/{dataset_id}", response_model=DatasetResponse)
@require_permission("update_dataset")
async def update_dataset(
    dataset_id: int,
    dataset_update: DatasetUpdate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update dataset information."""
    dataset_service = DatasetService(db)
    dataset = dataset_service.get_dataset_by_id(dataset_id)

    if not dataset or dataset.is_deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found"
        )

    if not (
        dataset.owner_id == current_user.id
        or require_permission("update_all_datasets")(current_user)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    updated_dataset = dataset_service.update_dataset_record(
        dataset_id,
        name=dataset_update.name,
        description=dataset_update.description,
        tags=dataset_update.tags,
        source=dataset_update.source,
        frequency=dataset_update.frequency,
    )

    if not updated_dataset:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update dataset",
        )

    AuditLogger.log_event(
        db=db,
        user_id=current_user.id,
        action="update_dataset",
        resource_type="dataset",
        resource_id=str(dataset_id),
        resource_name=updated_dataset.name,
        request=request,
    )
    logger.info(
        f"Dataset updated: {updated_dataset.name} by user {current_user.username}"
    )
    return DatasetResponse.from_orm(updated_dataset)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
@require_permission("delete_dataset")
async def delete_dataset(
    dataset_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Soft delete a dataset."""
    dataset_service = DatasetService(db)
    dataset = dataset_service.get_dataset_by_id(dataset_id)

    if not dataset or dataset.is_deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found"
        )

    if not (
        dataset.owner_id == current_user.id
        or require_permission("delete_all_datasets")(current_user)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    success = dataset_service.soft_delete_dataset(dataset_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete dataset",
        )

    if os.path.exists(dataset.file_path):
        try:
            os.remove(dataset.file_path)
            logger.info(f"Physical file deleted for dataset {dataset.id}")
        except Exception as e:
            logger.warning(
                f"Failed to delete physical file for dataset {dataset.id}: {e}"
            )

    AuditLogger.log_event(
        db=db,
        user_id=current_user.id,
        action="delete_dataset",
        resource_type="dataset",
        resource_id=str(dataset_id),
        resource_name=dataset.name,
        request=request,
    )
    logger.info(f"Dataset soft-deleted: {dataset.name} by user {current_user.username}")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ------------------------
# Dataset Analysis
# ------------------------
@router.get("/{dataset_id}/stats", response_model=DatasetStats)
@require_permission("read_dataset_stats")
async def get_dataset_statistics(
    dataset_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    encryption_manager: EncryptionManager = Depends(get_encryption_manager),
):
    """Get statistical information about dataset."""
    dataset_service = DatasetService(db)
    dataset = dataset_service.get_dataset_by_id(dataset_id)

    if not dataset or dataset.is_deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found"
        )

    if not (
        dataset.owner_id == current_user.id
        or require_permission("read_all_dataset_stats")(current_user)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    data = dataset_service.load_dataset_data(dataset.file_path, encryption_manager)
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load dataset data for statistics",
        )

    stats = dataset_service.calculate_dataset_statistics(data)
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate dataset statistics",
        )

    AuditLogger.log_event(
        db=db,
        user_id=current_user.id,
        action="read_dataset_stats",
        resource_type="dataset",
        resource_id=str(dataset_id),
        resource_name=dataset.name,
        request=request,
    )
    return DatasetStats(**stats)


@router.get("/{dataset_id}/preview")
@require_permission("read_dataset_preview")
async def preview_dataset(
    dataset_id: int,
    rows: int = Query(10, ge=1, le=100),
    request: Request = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    encryption_manager: EncryptionManager = Depends(get_encryption_manager),
    data_masking_manager: DataMaskingManager = Depends(get_data_masking_manager),
):
    """Get a preview of dataset data."""
    dataset_service = DatasetService(db)
    dataset = dataset_service.get_dataset_by_id(dataset_id)

    if not dataset or dataset.is_deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found"
        )

    if not (
        dataset.owner_id == current_user.id
        or require_permission("read_all_dataset_preview")(current_user)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    data = dataset_service.load_dataset_data(dataset.file_path, encryption_manager)
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load dataset data",
        )

    preview_data = data.head(rows)
    masked_records = [
        data_masking_manager.mask_object(record)
        for record in preview_data.to_dict("records")
    ]

    AuditLogger.log_event(
        db=db,
        user_id=current_user.id,
        action="read_dataset_preview",
        resource_type="dataset",
        resource_id=str(dataset_id),
        resource_name=dataset.name,
        request=request,
    )
    return {
        "columns": list(data.columns),
        "data": masked_records,
        "total_rows": len(data),
        "preview_rows": len(preview_data),
    }


@router.get("/{dataset_id}/download")
@require_permission("download_dataset")
async def download_dataset(
    dataset_id: int,
    format: str = Query("csv", pattern="^(csv|json|parquet)$"),
    request: Request = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    encryption_manager: EncryptionManager = Depends(get_encryption_manager),
):
    """Download dataset in specified format."""
    dataset_service = DatasetService(db)
    dataset = dataset_service.get_dataset_by_id(dataset_id)

    if not dataset or dataset.is_deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found"
        )

    if not (
        dataset.owner_id == current_user.id
        or require_permission("download_all_datasets")(current_user)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    data = dataset_service.load_dataset_data(dataset.file_path, encryption_manager)
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load dataset data",
        )

    # Temporary file for download
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}") as tmp_file:
        if format == "csv":
            data.to_csv(tmp_file.name, index=False)
            media_type = "text/csv"
        elif format == "json":
            data.to_json(tmp_file.name, orient="records", indent=2)
            media_type = "application/json"
        else:  # parquet
            data.to_parquet(tmp_file.name, index=False)
            media_type = "application/octet-stream"

        from fastapi.responses import FileResponse

        AuditLogger.log_event(
            db=db,
            user_id=current_user.id,
            action="download_dataset",
            resource_type="dataset",
            resource_id=str(dataset_id),
            resource_name=dataset.name,
            request=request,
        )
        return FileResponse(
            tmp_file.name, media_type=media_type, filename=f"{dataset.name}.{format}"
        )
