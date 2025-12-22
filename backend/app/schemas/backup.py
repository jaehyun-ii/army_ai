"""
Backup schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class BackupBase(BaseModel):
    """Base backup schema."""
    name: Optional[str] = None
    type: str = Field(default="manual", pattern="^(manual|scheduled)$")


class BackupCreate(BackupBase):
    """Schema for creating a backup."""
    pass


class BackupUpdate(BaseModel):
    """Schema for updating a backup."""
    status: Optional[str] = None
    db_file_path: Optional[str] = None
    storage_file_path: Optional[str] = None
    total_size_bytes: Optional[int] = None
    metadata_: Optional[Dict[str, Any]] = Field(default=None, alias="metadata")
    error_message: Optional[str] = None
    completed_at: Optional[datetime] = None
    restored_at: Optional[datetime] = None

    class Config:
        populate_by_name = True


class BackupInDB(BackupBase):
    """Schema for backup in database."""
    id: str
    status: str
    db_file_path: Optional[str] = None
    storage_file_path: Optional[str] = None
    total_size_bytes: Optional[int] = None
    metadata_: Optional[Dict[str, Any]] = Field(default=None, alias="metadata")
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    restored_at: Optional[datetime] = None

    class Config:
        from_attributes = True
        populate_by_name = True


class BackupResponse(BackupInDB):
    """Schema for backup response."""
    size: Optional[str] = None  # Human-readable size
    date: Optional[str] = None  # Formatted date string

    @classmethod
    def from_backup(cls, backup: "Backup"):  # type: ignore
        """Create response from backup model."""
        # Calculate human-readable size
        size_str = "N/A"
        if backup.total_size_bytes:
            size_bytes = backup.total_size_bytes
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            elif size_bytes < 1024 * 1024 * 1024:
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            else:
                size_str = f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

        # Format date
        date_str = backup.created_at.strftime("%Y-%m-%d %H:%M") if backup.created_at else None

        return cls(
            id=backup.id,
            name=backup.name,
            type=backup.type,
            status=backup.status,
            db_file_path=backup.db_file_path,
            storage_file_path=backup.storage_file_path,
            total_size_bytes=backup.total_size_bytes,
            metadata_=backup.metadata_,
            error_message=backup.error_message,
            created_at=backup.created_at,
            completed_at=backup.completed_at,
            restored_at=backup.restored_at,
            size=size_str,
            date=date_str
        )


class BackupListResponse(BaseModel):
    """Schema for list of backups."""
    backups: list[BackupResponse]
    total: int
    total_size_bytes: int
    total_size: str  # Human-readable total size
