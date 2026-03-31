"""
Dataset service for data management operations
"""

import os
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import and_
from sqlalchemy.orm import Session

from .. import models
from ..config import get_settings
from ..database import EncryptionManager

settings = get_settings()


class DatasetService:

    def __init__(self, db: Session) -> None:
        self.db = db

    def create_dataset_record(
        self,
        name: str,
        description: str,
        owner_id: int,
        file_path: Optional[str] = None,
        file_size: Optional[int] = None,
        file_hash: Optional[str] = None,
        columns_info: Optional[Dict] = None,
        row_count: Optional[int] = None,
        status: models.DatasetStatus = models.DatasetStatus.UPLOADING,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        frequency: Optional[str] = None,
    ) -> models.Dataset:
        """Create a new dataset record in the database."""
        owner = self.db.query(models.User).filter(models.User.id == owner_id).first()
        if not owner:
            raise ValueError("Owner not found")
        dataset = models.Dataset(
            name=name,
            description=description,
            owner_id=owner_id,
            file_path=file_path,
            file_size=file_size,
            file_hash=file_hash,
            columns_info=columns_info,
            row_count=row_count,
            status=status,
            tags=tags or [],
            source=source,
            frequency=frequency,
        )
        self.db.add(dataset)
        self.db.commit()
        self.db.refresh(dataset)
        return dataset

    def get_dataset_by_id(self, dataset_id: int) -> Optional[models.Dataset]:
        """Get dataset by ID"""
        return (
            self.db.query(models.Dataset)
            .filter(
                and_(models.Dataset.id == dataset_id, not models.Dataset.is_deleted)
            )
            .first()
        )

    def get_datasets_by_owner(
        self, owner_id: int, skip: int = 0, limit: int = 100
    ) -> List[models.Dataset]:
        """Get datasets by owner"""
        return (
            self.db.query(models.Dataset)
            .filter(
                and_(
                    models.Dataset.owner_id == owner_id,
                    not models.Dataset.is_deleted,
                )
            )
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_all_datasets(self, skip: int = 0, limit: int = 100) -> List[models.Dataset]:
        """Get all datasets (admin only)"""
        return (
            self.db.query(models.Dataset)
            .filter(not models.Dataset.is_deleted)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def update_dataset_record(
        self, dataset_id: int, **kwargs
    ) -> Optional[models.Dataset]:
        """Update dataset information"""
        dataset = self.get_dataset_by_id(dataset_id)
        if not dataset:
            return None
        for key, value in kwargs.items():
            if hasattr(dataset, key) and key not in [
                "id",
                "owner_id",
                "created_at",
                "file_path",
                "file_size",
                "file_hash",
                "columns_info",
                "row_count",
                "status",
            ]:
                setattr(dataset, key, value)
        self.db.commit()
        self.db.refresh(dataset)
        return dataset

    def soft_delete_dataset(self, dataset_id: int, deleted_by_id: int) -> bool:
        """Soft delete dataset"""
        dataset = self.get_dataset_by_id(dataset_id)
        if not dataset:
            return False
        dataset.is_deleted = True
        dataset.deleted_at = datetime.utcnow()
        dataset.deleted_by_id = deleted_by_id
        self.db.commit()
        return True

    def load_dataset_data(
        self, file_path: str, encryption_manager: EncryptionManager
    ) -> Optional[pd.DataFrame]:
        """Load dataset data as pandas DataFrame, with decryption if enabled"""
        if not os.path.exists(file_path):
            return None
        try:
            if settings.compliance.enable_data_encryption:
                with open(file_path, "r", encoding="utf-8") as f:
                    encrypted_content = f.read()
                decrypted_content = encryption_manager.decrypt(encrypted_content)
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".tmp"
                ) as tmp_file:
                    tmp_file.write(decrypted_content.encode("utf-8"))
                    tmp_file_path = tmp_file.name
            else:
                tmp_file_path = file_path
            if tmp_file_path.endswith(".csv") or tmp_file_path.endswith(".tmp"):
                return pd.read_csv(tmp_file_path)
            elif tmp_file_path.endswith(".json"):
                return pd.read_json(tmp_file_path)
            elif tmp_file_path.endswith(tuple(settings.allowed_file_types)):
                if tmp_file_path.endswith((".xlsx", ".xls")):
                    return pd.read_excel(tmp_file_path)
                else:
                    return None
            else:
                return None
        except Exception as e:
            logger.error(f"Error loading or decrypting dataset from {file_path}: {e}")
            return None
        finally:
            if (
                settings.compliance.enable_data_encryption
                and os.path.exists(tmp_file_path)
                and (tmp_file_path != file_path)
            ):
                os.remove(tmp_file_path)

    def calculate_dataset_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistical information about a pandas DataFrame"""
        stats = {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "missing_values": data.isnull().sum().to_dict(),
            "numeric_stats": (
                data.describe().to_dict()
                if len(data.select_dtypes(include=["number"]).columns) > 0
                else {}
            ),
            "categorical_stats": {},
        }
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            stats["categorical_stats"][col] = {
                "unique_count": data[col].nunique(),
                "top_values": data[col].value_counts().head(10).to_dict(),
            }
        return stats
