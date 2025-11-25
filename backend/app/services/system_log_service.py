"""
System log service for managing system logs.
"""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_, and_, desc
from sqlalchemy.orm import selectinload

from app.models.system_log import SystemLog, LogLevel

logger = logging.getLogger(__name__)


class SystemLogService:
    """Service for managing system logs."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_log(
        self,
        action: str,
        message: str,
        log_level: LogLevel = LogLevel.INFO,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        module: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        status_code: Optional[int] = None,
        response_time_ms: Optional[int] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        stack_trace: Optional[str] = None,
    ) -> SystemLog:
        """
        Create a new system log entry.

        Args:
            action: Action being logged
            message: Log message
            log_level: Log severity level
            user_id: User ID (if applicable)
            username: Username (if applicable)
            ip_address: IP address of the request
            user_agent: User agent string
            module: Module name
            details: Additional details as JSON
            request_id: Request ID for tracing
            endpoint: API endpoint
            method: HTTP method
            status_code: HTTP status code
            response_time_ms: Response time in milliseconds
            error_type: Error type (for errors)
            error_message: Error message (for errors)
            stack_trace: Stack trace (for errors)

        Returns:
            Created SystemLog instance
        """
        try:
            log = SystemLog(
                log_level=log_level,
                user_id=user_id,
                username=username,
                ip_address=ip_address,
                user_agent=user_agent,
                action=action,
                module=module,
                message=message,
                details=details,
                request_id=request_id,
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                response_time_ms=response_time_ms,
                error_type=error_type,
                error_message=error_message,
                stack_trace=stack_trace,
            )
            self.db.add(log)
            await self.db.commit()
            await self.db.refresh(log)
            return log
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to create system log: {e}")
            raise

    async def get_logs(
        self,
        log_level: Optional[LogLevel] = None,
        user_id: Optional[str] = None,
        module: Optional[str] = None,
        action: Optional[str] = None,
        search_term: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[List[SystemLog], int]:
        """
        Get system logs with filtering and pagination.

        Args:
            log_level: Filter by log level
            user_id: Filter by user ID
            module: Filter by module
            action: Filter by action
            search_term: Search in message, username, action
            start_date: Filter logs after this date
            end_date: Filter logs before this date
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            Tuple of (logs list, total count)
        """
        try:
            # Build base query
            query = select(SystemLog)

            # Apply filters
            conditions = []

            if log_level:
                conditions.append(SystemLog.log_level == log_level)

            if user_id:
                conditions.append(SystemLog.user_id == user_id)

            if module:
                conditions.append(SystemLog.module == module)

            if action:
                conditions.append(SystemLog.action.ilike(f"%{action}%"))

            if search_term:
                search_conditions = [
                    SystemLog.message.ilike(f"%{search_term}%"),
                    SystemLog.username.ilike(f"%{search_term}%"),
                    SystemLog.action.ilike(f"%{search_term}%"),
                ]
                conditions.append(or_(*search_conditions))

            if start_date:
                conditions.append(SystemLog.timestamp >= start_date)

            if end_date:
                conditions.append(SystemLog.timestamp <= end_date)

            if conditions:
                query = query.where(and_(*conditions))

            # Get total count
            count_query = select(func.count()).select_from(SystemLog)
            if conditions:
                count_query = count_query.where(and_(*conditions))
            result = await self.db.execute(count_query)
            total = result.scalar_one()

            # Apply pagination and ordering
            query = query.order_by(desc(SystemLog.timestamp)).offset(skip).limit(limit)

            # Load user relationship
            query = query.options(selectinload(SystemLog.user))

            # Execute query
            result = await self.db.execute(query)
            logs = result.scalars().all()

            return list(logs), total

        except Exception as e:
            logger.error(f"Failed to get system logs: {e}")
            raise

    async def get_log_by_id(self, log_id: str) -> Optional[SystemLog]:
        """
        Get a specific log by ID.

        Args:
            log_id: Log ID

        Returns:
            SystemLog instance or None
        """
        try:
            query = select(SystemLog).where(SystemLog.id == log_id).options(selectinload(SystemLog.user))
            result = await self.db.execute(query)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get log {log_id}: {e}")
            raise

    async def get_log_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get log statistics (counts by level, recent errors, etc.).

        Args:
            start_date: Start date for statistics
            end_date: End date for statistics

        Returns:
            Dictionary with statistics
        """
        try:
            # Default to last 24 hours if no dates provided
            if not start_date:
                start_date = datetime.utcnow() - timedelta(hours=24)
            if not end_date:
                end_date = datetime.utcnow()

            # Build base conditions
            conditions = [
                SystemLog.timestamp >= start_date,
                SystemLog.timestamp <= end_date,
            ]

            # Get counts by level
            stats = {}
            for level in LogLevel:
                count_query = select(func.count()).select_from(SystemLog).where(
                    and_(SystemLog.log_level == level, *conditions)
                )
                result = await self.db.execute(count_query)
                stats[level.value.lower()] = result.scalar_one()

            # Get total count
            total_query = select(func.count()).select_from(SystemLog).where(and_(*conditions))
            result = await self.db.execute(total_query)
            stats["total"] = result.scalar_one()

            # Get recent errors (last 10)
            error_query = (
                select(SystemLog)
                .where(and_(SystemLog.log_level.in_([LogLevel.ERROR, LogLevel.CRITICAL]), *conditions))
                .order_by(desc(SystemLog.timestamp))
                .limit(10)
            )
            result = await self.db.execute(error_query)
            recent_errors = result.scalars().all()
            stats["recent_errors"] = [log.to_dict() for log in recent_errors]

            return stats

        except Exception as e:
            logger.error(f"Failed to get log statistics: {e}")
            raise

    async def delete_logs(
        self,
        before_date: Optional[datetime] = None,
        log_level: Optional[LogLevel] = None,
    ) -> int:
        """
        Delete logs based on criteria.

        Args:
            before_date: Delete logs before this date
            log_level: Delete logs with this level

        Returns:
            Number of deleted logs
        """
        try:
            conditions = []

            if before_date:
                conditions.append(SystemLog.timestamp < before_date)

            if log_level:
                conditions.append(SystemLog.log_level == log_level)

            if not conditions:
                raise ValueError("At least one deletion criteria must be provided")

            # Count logs to be deleted
            count_query = select(func.count()).select_from(SystemLog).where(and_(*conditions))
            result = await self.db.execute(count_query)
            count = result.scalar_one()

            # Delete logs
            from sqlalchemy import delete

            delete_query = delete(SystemLog).where(and_(*conditions))
            await self.db.execute(delete_query)
            await self.db.commit()

            logger.info(f"Deleted {count} system logs")
            return count

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to delete logs: {e}")
            raise


# Helper function to create log service
def get_system_log_service(db: AsyncSession) -> SystemLogService:
    """Get system log service instance."""
    return SystemLogService(db)
