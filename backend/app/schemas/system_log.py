"""
System log schemas.
"""
from pydantic import BaseModel, ConfigDict, field_serializer
from typing import Optional, Dict, Any, Union
from datetime import datetime
from uuid import UUID
from ipaddress import IPv4Address, IPv6Address

from app.models.system_log import LogLevel


class SystemLogBase(BaseModel):
    """Base system log schema."""

    log_level: LogLevel
    action: str
    message: str
    module: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class SystemLogCreate(SystemLogBase):
    """Schema for creating a system log."""

    user_id: Optional[UUID] = None
    username: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    status_code: Optional[int] = None
    response_time_ms: Optional[int] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None


class SystemLogResponse(SystemLogBase):
    """Schema for system log response."""

    id: UUID
    timestamp: datetime
    user_id: Optional[UUID] = None
    username: Optional[str] = None
    ip_address: Optional[Union[str, IPv4Address, IPv6Address]] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    status_code: Optional[int] = None
    response_time_ms: Optional[int] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @field_serializer('ip_address')
    def serialize_ip_address(self, ip: Optional[Union[str, IPv4Address, IPv6Address]]) -> Optional[str]:
        """Convert IPv4Address/IPv6Address to string."""
        if ip is None:
            return None
        return str(ip)


class SystemLogFilter(BaseModel):
    """Schema for filtering system logs."""

    log_level: Optional[LogLevel] = None
    user_id: Optional[UUID] = None
    module: Optional[str] = None
    action: Optional[str] = None
    search_term: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    skip: int = 0
    limit: int = 100


class SystemLogStatistics(BaseModel):
    """Schema for system log statistics."""

    total: int
    debug: int
    info: int
    warning: int
    error: int
    critical: int
    recent_errors: list[Dict[str, Any]]


class SystemLogListResponse(BaseModel):
    """Schema for paginated system log list response."""

    logs: list[SystemLogResponse]
    total: int
    skip: int
    limit: int
