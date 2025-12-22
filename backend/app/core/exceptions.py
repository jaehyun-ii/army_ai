"""
Unified exception handling for the application.
"""
from typing import Any, Dict, Optional
from fastapi import HTTPException, status


class BaseAPIException(HTTPException):
    """Base exception for all API exceptions."""

    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: str = "An error occurred",
        headers: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code or self.__class__.__name__


class ValidationError(BaseAPIException):
    """Raised when input validation fails."""

    def __init__(self, detail: str = "Validation failed", **kwargs):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="VALIDATION_ERROR",
            **kwargs
        )


class NotFoundError(BaseAPIException):
    """Raised when a resource is not found."""

    def __init__(self, resource: str = "Resource", **kwargs):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} not found",
            error_code="NOT_FOUND",
            **kwargs
        )


class ConflictError(BaseAPIException):
    """Raised when there's a conflict with existing data."""

    def __init__(self, detail: str = "Conflict with existing data", **kwargs):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            error_code="CONFLICT",
            **kwargs
        )


class UnauthorizedError(BaseAPIException):
    """Raised when authentication fails."""

    def __init__(self, detail: str = "Unauthorized", **kwargs):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code="UNAUTHORIZED",
            headers={"WWW-Authenticate": "Bearer"},
            **kwargs
        )


class ForbiddenError(BaseAPIException):
    """Raised when user lacks permissions."""

    def __init__(self, detail: str = "Forbidden", **kwargs):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code="FORBIDDEN",
            **kwargs
        )


class RateLimitError(BaseAPIException):
    """Raised when rate limit is exceeded."""

    def __init__(self, detail: str = "Rate limit exceeded", retry_after: int = 60, **kwargs):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            error_code="RATE_LIMIT_EXCEEDED",
            headers={"Retry-After": str(retry_after)},
            **kwargs
        )


class ServiceUnavailableError(BaseAPIException):
    """Raised when a service is temporarily unavailable."""

    def __init__(self, detail: str = "Service temporarily unavailable", **kwargs):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            error_code="SERVICE_UNAVAILABLE",
            **kwargs
        )


class BadGatewayError(BaseAPIException):
    """Raised when an upstream service returns an invalid response."""

    def __init__(self, detail: str = "Bad gateway", **kwargs):
        super().__init__(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail,
            error_code="BAD_GATEWAY",
            **kwargs
        )


class PayloadTooLargeError(BaseAPIException):
    """Raised when the request payload is too large."""

    def __init__(self, max_size: int, **kwargs):
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Payload too large. Maximum size: {max_size} bytes",
            error_code="PAYLOAD_TOO_LARGE",
            **kwargs
        )


class InternalServerError(BaseAPIException):
    """Raised for internal server errors."""

    def __init__(self, detail: str = "Internal server error", **kwargs):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="INTERNAL_SERVER_ERROR",
            **kwargs
        )


# Domain-specific exceptions
class ModelNotFoundError(NotFoundError):
    """Raised when a model is not found."""

    def __init__(self, model_id: str, **kwargs):
        super().__init__(resource=f"Model {model_id}", **kwargs)
        self.error_code = "MODEL_NOT_FOUND"


class DatasetNotFoundError(NotFoundError):
    """Raised when a dataset is not found."""

    def __init__(self, dataset_id: str, **kwargs):
        super().__init__(resource=f"Dataset {dataset_id}", **kwargs)
        self.error_code = "DATASET_NOT_FOUND"


class InvalidImageError(ValidationError):
    """Raised when an image is invalid."""

    def __init__(self, detail: str = "Invalid image format or corrupted file", **kwargs):
        super().__init__(detail=detail, **kwargs)
        self.error_code = "INVALID_IMAGE"


class ModelLoadError(InternalServerError):
    """Raised when a model fails to load."""

    def __init__(self, model_path: str, detail: str = None, **kwargs):
        detail = detail or f"Failed to load model from {model_path}"
        super().__init__(detail=detail, **kwargs)
        self.error_code = "MODEL_LOAD_ERROR"


class AttackFailedError(InternalServerError):
    """Raised when an adversarial attack fails."""

    def __init__(self, attack_type: str, detail: str = None, **kwargs):
        detail = detail or f"{attack_type} attack failed"
        super().__init__(detail=detail, **kwargs)
        self.error_code = "ATTACK_FAILED"


class StorageError(InternalServerError):
    """Raised when storage operations fail."""

    def __init__(self, operation: str, detail: str = None, **kwargs):
        detail = detail or f"Storage {operation} failed"
        super().__init__(detail=detail, **kwargs)
        self.error_code = "STORAGE_ERROR"


class DatabaseError(InternalServerError):
    """Raised when database operations fail."""

    def __init__(self, operation: str, detail: str = None, **kwargs):
        detail = detail or f"Database {operation} failed"
        super().__init__(detail=detail, **kwargs)
        self.error_code = "DATABASE_ERROR"