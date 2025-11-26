import time
import uuid
import jwt
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from app.database import AsyncSessionLocal
from app.services.system_log_service import SystemLogService
from app.models.system_log import LogLevel
from app.core.config import settings

logger = logging.getLogger(__name__)

class SystemLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all API requests to the database.
    """
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Skip logging for health checks, docs, static files, and system-logs endpoints
        # (to prevent recursive logging - viewing logs shouldn't create more logs)
        if request.url.path.startswith("/health") or \
           request.url.path.startswith("/docs") or \
           request.url.path.startswith("/openapi.json") or \
           request.url.path.startswith("/redoc") or \
           request.url.path.startswith("/storage") or \
           request.url.path.startswith(f"{settings.API_PREFIX}/system-logs") or \
           not request.url.path.startswith(settings.API_PREFIX):
            return await call_next(request)

        response = await call_next(request)
        
        process_time = (time.time() - start_time) * 1000
        
        # Extract user info from token if present
        user_id = None
        username = None

        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
                # JWT payload structure:
                # - sub: user ID (UUID as string)
                # - username: user's username
                # - email: user's email
                # - role: user's role
                user_id = payload.get("sub")  # User ID is in 'sub' field
                username = payload.get("username")  # Actual username
            except jwt.ExpiredSignatureError:
                # Token expired - treat as anonymous but log for monitoring
                logger.debug(f"Expired token in request to {request.url.path}")
            except jwt.InvalidTokenError:
                # Invalid token - treat as anonymous
                logger.debug(f"Invalid token in request to {request.url.path}")
            except Exception as e:
                # Unexpected error during token decoding
                logger.warning(f"Unexpected error decoding token: {type(e).__name__}: {str(e)}")

        # Log to database
        try:
            async with AsyncSessionLocal() as db:
                service = SystemLogService(db)

                log_level = LogLevel.INFO
                if response.status_code >= 500:
                    log_level = LogLevel.ERROR
                elif response.status_code >= 400:
                    log_level = LogLevel.WARNING

                await service.create_log(
                    action=f"{request.method} {request.url.path}",
                    message=f"Request to {request.url.path} completed with status {response.status_code}",
                    log_level=log_level,
                    user_id=user_id,
                    username=username,
                    ip_address=request.client.host if request.client else None,
                    user_agent=request.headers.get("user-agent"),
                    module="api",
                    request_id=request_id,
                    endpoint=request.url.path,
                    method=request.method,
                    status_code=response.status_code,
                    response_time_ms=int(process_time)
                )
        except ConnectionError as e:
            # Database connection error
            logger.error(f"Database connection failed during logging: {str(e)}")
        except Exception as e:
            # Don't fail the request if logging fails
            # Log error for monitoring but don't crash the application
            logger.error(f"Failed to log request to database: {type(e).__name__}: {str(e)}", exc_info=True)

        return response
