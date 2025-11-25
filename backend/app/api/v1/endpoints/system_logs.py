"""
System logs endpoints for admin log management.
"""
from typing import Optional
from uuid import UUID
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.system_log import (
    SystemLogResponse,
    SystemLogListResponse,
    SystemLogStatistics,
    SystemLogCreate,
)
from app.models.system_log import LogLevel
from app.models.user import User
from app.services.auth_service import get_current_active_admin
from app.services.system_log_service import SystemLogService

router = APIRouter()


@router.get("/", response_model=SystemLogListResponse)
async def get_system_logs(
    log_level: Optional[LogLevel] = Query(None, description="Filter by log level"),
    user_id: Optional[UUID] = Query(None, description="Filter by user ID"),
    module: Optional[str] = Query(None, description="Filter by module"),
    action: Optional[str] = Query(None, description="Filter by action"),
    search_term: Optional[str] = Query(None, description="Search in message, username, action"),
    start_date: Optional[datetime] = Query(None, description="Filter logs after this date"),
    end_date: Optional[datetime] = Query(None, description="Filter logs before this date"),
    hours: Optional[int] = Query(None, description="Get logs from last N hours (overrides start_date)"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin),
):
    """
    Get system logs with filtering and pagination.

    Only accessible by admin users.

    Args:
        log_level: Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        user_id: Filter by user ID
        module: Filter by module name
        action: Filter by action (supports partial match)
        search_term: Search term for message, username, and action
        start_date: Get logs after this date
        end_date: Get logs before this date
        hours: Get logs from last N hours (convenience parameter)
        skip: Number of records to skip for pagination
        limit: Maximum number of records to return

    Returns:
        Paginated list of system logs with total count
    """
    try:
        # If hours parameter is provided, calculate start_date
        if hours is not None:
            start_date = datetime.utcnow() - timedelta(hours=hours)

        service = SystemLogService(db)
        logs, total = await service.get_logs(
            log_level=log_level,
            user_id=user_id,
            module=module,
            action=action,
            search_term=search_term,
            start_date=start_date,
            end_date=end_date,
            skip=skip,
            limit=limit,
        )

        return SystemLogListResponse(
            logs=logs,
            total=total,
            skip=skip,
            limit=limit,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system logs: {str(e)}",
        )


@router.get("/statistics", response_model=SystemLogStatistics)
async def get_log_statistics(
    start_date: Optional[datetime] = Query(None, description="Start date for statistics"),
    end_date: Optional[datetime] = Query(None, description="End date for statistics"),
    hours: Optional[int] = Query(24, description="Get statistics from last N hours (default: 24)"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin),
):
    """
    Get system log statistics (counts by level, recent errors, etc.).

    Only accessible by admin users.

    Args:
        start_date: Start date for statistics
        end_date: End date for statistics
        hours: Get statistics from last N hours (default: 24)

    Returns:
        Log statistics including counts by level and recent errors
    """
    try:
        # If hours parameter is provided and start_date is not, calculate start_date
        if hours is not None and start_date is None:
            start_date = datetime.utcnow() - timedelta(hours=hours)

        service = SystemLogService(db)
        stats = await service.get_log_statistics(
            start_date=start_date,
            end_date=end_date,
        )

        return SystemLogStatistics(**stats)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve log statistics: {str(e)}",
        )


@router.get("/{log_id}", response_model=SystemLogResponse)
async def get_log_by_id(
    log_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin),
):
    """
    Get a specific log entry by ID.

    Only accessible by admin users.

    Args:
        log_id: Log entry ID

    Returns:
        System log entry
    """
    try:
        service = SystemLogService(db)
        log = await service.get_log_by_id(str(log_id))

        if not log:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Log entry not found",
            )

        return log

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve log entry: {str(e)}",
        )


@router.post("/", response_model=SystemLogResponse, status_code=status.HTTP_201_CREATED)
async def create_log_entry(
    log_data: SystemLogCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin),
):
    """
    Manually create a system log entry.

    Only accessible by admin users. Use this for manual logging or external system integration.

    Args:
        log_data: Log entry data

    Returns:
        Created system log entry
    """
    try:
        service = SystemLogService(db)
        log = await service.create_log(
            action=log_data.action,
            message=log_data.message,
            log_level=log_data.log_level,
            user_id=str(log_data.user_id) if log_data.user_id else None,
            username=log_data.username,
            ip_address=log_data.ip_address,
            user_agent=log_data.user_agent,
            module=log_data.module,
            details=log_data.details,
            request_id=log_data.request_id,
            endpoint=log_data.endpoint,
            method=log_data.method,
            status_code=log_data.status_code,
            response_time_ms=log_data.response_time_ms,
            error_type=log_data.error_type,
            error_message=log_data.error_message,
            stack_trace=log_data.stack_trace,
        )

        return log

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create log entry: {str(e)}",
        )


@router.delete("/cleanup")
async def cleanup_logs(
    before_date: Optional[datetime] = Query(None, description="Delete logs before this date"),
    days: Optional[int] = Query(None, description="Delete logs older than N days"),
    log_level: Optional[LogLevel] = Query(None, description="Delete logs with specific level"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin),
):
    """
    Delete old system logs based on criteria.

    Only accessible by admin users. Use with caution!

    Args:
        before_date: Delete logs before this date
        days: Delete logs older than N days (convenience parameter)
        log_level: Delete logs with this specific level

    Returns:
        Number of deleted logs
    """
    try:
        # If days parameter is provided, calculate before_date
        if days is not None:
            before_date = datetime.utcnow() - timedelta(days=days)

        if not before_date and not log_level:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'before_date', 'days', or 'log_level' must be provided",
            )

        service = SystemLogService(db)
        deleted_count = await service.delete_logs(
            before_date=before_date,
            log_level=log_level,
        )

        return {
            "message": f"Successfully deleted {deleted_count} log entries",
            "deleted_count": deleted_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete logs: {str(e)}",
        )
