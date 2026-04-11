"""
File handling service for dataset uploads and model storage
"""

import hashlib
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..domain import models

settings = get_settings()


class FileService:
    """Service for handling file uploads and storage"""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.upload_dir = Path(settings.upload_directory)
        self.model_dir = Path(settings.model_directory)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    async def upload_dataset(
        self, file: UploadFile, user_id: int, name: str, description: str = None
    ) -> models.Dataset:
        """Upload and process a dataset file"""
        self._validate_file(file)
        file_hash = await self._calculate_file_hash(file)
        file_extension = Path(file.filename).suffix
        unique_filename = f"{file_hash}{file_extension}"
        file_path = self.upload_dir / unique_filename
        try:
            await self._save_file(file, file_path)
            dataset_info = await self._process_dataset(file_path)
            dataset = models.Dataset(
                name=name,
                description=description,
                owner_id=user_id,
                file_path=str(file_path),
                file_size=file_path.stat().st_size,
                file_type=file_extension,
                columns=dataset_info["columns"],
                row_count=dataset_info["row_count"],
                metadata=dataset_info["metadata"],
            )
            self.db.add(dataset)
            self.db.commit()
            self.db.refresh(dataset)
            return dataset
        except Exception as e:
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(
                status_code=400, detail=f"Failed to process dataset: {str(e)}"
            )

    def _validate_file(self, file: UploadFile) -> Any:
        """Validate uploaded file"""
        if hasattr(file, "size") and file.size > settings.max_upload_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {settings.max_upload_size / (1024 * 1024):.1f}MB",
            )
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in settings.allowed_file_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {', '.join(settings.allowed_file_types)}",
            )
        mime_type, _ = mimetypes.guess_type(file.filename)
        allowed_mime_types = [
            "text/csv",
            "application/json",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ]
        if mime_type and mime_type not in allowed_mime_types:
            raise HTTPException(
                status_code=400, detail=f"MIME type not allowed: {mime_type}"
            )

    async def _calculate_file_hash(self, file: UploadFile) -> str:
        """Calculate SHA-256 hash of file content"""
        hasher = hashlib.sha256()
        await file.seek(0)
        while chunk := (await file.read(8192)):
            hasher.update(chunk)
        await file.seek(0)
        return hasher.hexdigest()

    async def _save_file(self, file: UploadFile, file_path: Path):
        """Save uploaded file to disk"""
        try:
            with open(file_path, "wb") as buffer:
                while chunk := (await file.read(8192)):
                    buffer.write(chunk)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to save file: {str(e)}"
            )

    async def _process_dataset(self, file_path: Path) -> Dict[str, Any]:
        """Process and analyze dataset file"""
        try:
            file_extension = file_path.suffix.lower()
            if file_extension == ".csv":
                dataframe = pd.read_csv(file_path)
            elif file_extension == ".json":
                dataframe = pd.read_json(file_path)
            elif file_extension in [".xlsx", ".xls"]:
                dataframe = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            columns = list(dataframe.columns)
            row_count = len(dataframe)
            metadata = {
                "column_types": dataframe.dtypes.astype(str).to_dict(),
                "null_counts": dataframe.isnull().sum().to_dict(),
                "numeric_columns": dataframe.select_dtypes(
                    include=["number"]
                ).columns.tolist(),
                "categorical_columns": dataframe.select_dtypes(
                    include=["object"]
                ).columns.tolist(),
                "datetime_columns": dataframe.select_dtypes(
                    include=["datetime"]
                ).columns.tolist(),
                "memory_usage": dataframe.memory_usage(deep=True).sum(),
                "shape": dataframe.shape,
                "sample_data": (
                    dataframe.head(5).to_dict("records") if row_count > 0 else []
                ),
            }
            if metadata["numeric_columns"]:
                numeric_stats = (
                    dataframe[metadata["numeric_columns"]].describe().to_dict()
                )
                metadata["numeric_statistics"] = numeric_stats
            return {"columns": columns, "row_count": row_count, "metadata": metadata}
        except Exception as e:
            raise ValueError(f"Failed to process dataset: {str(e)}")

    def get_dataset_preview(self, dataset_id: int, rows: int = 10) -> Dict[str, Any]:
        """Get preview of dataset"""
        dataset = (
            self.db.query(models.Dataset)
            .filter(models.Dataset.id == dataset_id)
            .first()
        )
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        try:
            file_path = Path(dataset.file_path)
            file_extension = file_path.suffix.lower()
            if file_extension == ".csv":
                dataframe = pd.read_csv(file_path, nrows=rows)
            elif file_extension == ".json":
                dataframe = pd.read_json(file_path)
                dataframe = dataframe.head(rows)
            elif file_extension in [".xlsx", ".xls"]:
                dataframe = pd.read_excel(file_path, nrows=rows)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            return {
                "columns": list(dataframe.columns),
                "data": dataframe.to_dict("records"),
                "total_rows": dataset.row_count,
                "preview_rows": len(dataframe),
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to preview dataset: {str(e)}"
            )

    def delete_dataset_file(self, dataset_id: int) -> bool:
        """Delete dataset file from disk"""
        dataset = (
            self.db.query(models.Dataset)
            .filter(models.Dataset.id == dataset_id)
            .first()
        )
        if not dataset:
            return False
        try:
            file_path = Path(dataset.file_path)
            if file_path.exists():
                file_path.unlink()
            return True
        except Exception:
            return False

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get information about a file"""
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise HTTPException(status_code=404, detail="File not found")
        stat = file_path_obj.stat()
        return {
            "filename": file_path_obj.name,
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": file_path_obj.suffix,
            "mime_type": mimetypes.guess_type(str(file_path_obj))[0],
        }

    def cleanup_old_files(self, days: int = 30) -> int:
        """Clean up old uploaded files"""
        cutoff_time = datetime.now().timestamp() - days * 24 * 60 * 60
        deleted_count = 0
        for file_path_obj in self.upload_dir.iterdir():
            if file_path_obj.is_file() and file_path_obj.stat().st_mtime < cutoff_time:
                dataset = (
                    self.db.query(models.Dataset)
                    .filter(models.Dataset.file_path == str(file_path_obj))
                    .first()
                )
                if not dataset:
                    try:
                        file_path_obj.unlink()
                        deleted_count += 1
                    except Exception:
                        pass
        return deleted_count

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        upload_size = sum(
            (
                file_obj.stat().st_size
                for file_obj in self.upload_dir.rglob("*")
                if file_obj.is_file()
            )
        )
        model_size = sum(
            (
                file_obj.stat().st_size
                for file_obj in self.model_dir.rglob("*")
                if file_obj.is_file()
            )
        )
        upload_count = len(
            [file_obj for file_obj in self.upload_dir.rglob("*") if file_obj.is_file()]
        )
        model_count = len(
            [file_obj for file_obj in self.model_dir.rglob("*") if file_obj.is_file()]
        )
        return {
            "upload_directory": {
                "path": str(self.upload_dir),
                "size_bytes": upload_size,
                "size_mb": upload_size / (1024 * 1024),
                "file_count": upload_count,
            },
            "model_directory": {
                "path": str(self.model_dir),
                "size_bytes": model_size,
                "size_mb": model_size / (1024 * 1024),
                "file_count": model_count,
            },
            "total_size_bytes": upload_size + model_size,
            "total_size_mb": (upload_size + model_size) / (1024 * 1024),
        }
