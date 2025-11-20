"""
Enhanced configuration management with environment-specific settings.
"""
import os
import secrets
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, ValidationInfo


class Environment(str, Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class Settings(BaseSettings):
    """Application settings with environment-specific configurations."""

    # Environment
    ENVIRONMENT: Environment = Environment.DEVELOPMENT

    # Database
    DATABASE_URL: str
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_POOL_PRE_PING: bool = True
    DATABASE_ECHO: bool = False

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str, info: ValidationInfo) -> str:
        """Validate DATABASE_URL is appropriate for environment."""
        env = info.data.get("ENVIRONMENT", Environment.DEVELOPMENT)

        if env == Environment.PRODUCTION:
            # Ensure production doesn't use localhost or SQLite
            if any(x in v.lower() for x in ['localhost', '127.0.0.1', 'sqlite']):
                raise ValueError(
                    "⚠️  Production environment cannot use localhost or SQLite database.\n"
                    "Please configure a proper PostgreSQL connection."
                )

        return v

    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    PASSWORD_MIN_LENGTH: int = 8

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v: str, info: ValidationInfo) -> str:
        """Validate SECRET_KEY is secure."""
        env = info.data.get("ENVIRONMENT", Environment.DEVELOPMENT)

        if v == "your-secret-key-here-change-in-production":
            if env == Environment.PRODUCTION:
                raise ValueError(
                    "⚠️  DEFAULT SECRET_KEY IN PRODUCTION! This is a security risk.\n"
                    "Generate a secure key with:\n"
                    "  python -c 'import secrets; print(secrets.token_urlsafe(32))'"
                )
            # Auto-generate for development/testing
            return secrets.token_urlsafe(32)

        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters for security")
        return v

    # Application
    APP_NAME: str = "Adversarial Vision Platform API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_PREFIX: str = "/api/v1"

    @field_validator("DEBUG")
    @classmethod
    def validate_debug(cls, v: bool, info: ValidationInfo) -> bool:
        """Ensure DEBUG is False in production."""
        env = info.data.get("ENVIRONMENT", Environment.DEVELOPMENT)
        if env == Environment.PRODUCTION and v:
            raise ValueError("DEBUG must be False in production")
        return v

    # Storage
    STORAGE_ROOT: str = "./storage"
    STORAGE_TYPE: str = "local"  # local or s3
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/bmp"]
    DEFAULT_PATCH_PLUGIN: str = "global_pgd_2d"
    DEFAULT_NOISE_PLUGIN: str = "fgsm_2d"
    MODEL_WEIGHT_EXTENSIONS: List[str] = [
        ".pt",
        ".pth",
        ".weights",
        ".onnx",
        ".pb",
        ".h5",
        ".tflite",
    ]

    # Image format mappings (single source of truth)
    IMAGE_FORMAT_MAP: dict = {
        "image/jpeg": [".jpg", ".jpeg"],
        "image/png": [".png"],
        "image/bmp": [".bmp"],
        # Additional formats can be enabled by adding to ALLOWED_IMAGE_TYPES
        "image/tiff": [".tiff", ".tif"],
        "image/webp": [".webp"],
    }

    # S3 (optional)
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: Optional[str] = None
    S3_BUCKET: Optional[str] = None

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:3003",
        "http://localhost:8080",
        "http://localhost:8000",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8000",
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    # Third-party library logging levels (configurable via environment)
    # Format: LOGGER_NAME=LEVEL (e.g., LOG_THIRD_PARTY_LEVELS='{"uvicorn.access": "WARNING", "sqlalchemy.engine": "ERROR"}')
    LOG_THIRD_PARTY_LEVELS: Optional[dict] = None

    @field_validator("LOG_THIRD_PARTY_LEVELS", mode="before")
    @classmethod
    def parse_log_levels(cls, v: Union[str, dict, None]) -> Optional[dict]:
        """Parse third-party log levels from string or dict."""
        if v is None:
            # Default third-party log levels
            return {
                "uvicorn.access": "WARNING",
                "sqlalchemy.engine": "WARNING",
                "asyncio": "WARNING",
                "PIL": "ERROR",
                "urllib3": "ERROR",
            }
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # Fallback to default
                return {
                    "uvicorn.access": "WARNING",
                    "sqlalchemy.engine": "WARNING",
                    "asyncio": "WARNING",
                }
        return v

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60

    # Cache
    CACHE_TYPE: str = "memory"  # memory, redis
    REDIS_URL: Optional[str] = None
    CACHE_TTL: int = 300  # 5 minutes

    # AI/ML Settings
    MODEL_CACHE_SIZE: int = 5
    MAX_BATCH_SIZE: int = 32
    GPU_MEMORY_FRACTION: float = 0.8
    DEFAULT_INPUT_SIZE: int = 640

    # Performance
    WORKER_CONNECTIONS: int = 1000
    KEEPALIVE: int = 5

    model_config = SettingsConfigDict(
        env_file=".env" if os.getenv("ENVIRONMENT") != "testing" else ".env.test",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.ENVIRONMENT == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if running in development."""
        return self.ENVIRONMENT == Environment.DEVELOPMENT

    def get_storage_path(self, *paths: str) -> Path:
        """Get storage path."""
        base_path = Path(self.STORAGE_ROOT)
        if paths:
            return base_path.joinpath(*paths)
        return base_path

    def get_allowed_image_extensions(self) -> set:
        """Get allowed image file extensions based on ALLOWED_IMAGE_TYPES.

        Returns:
            Set of allowed file extensions (e.g., {'.jpg', '.jpeg', '.png', '.bmp'})
        """
        extensions = set()
        for mime_type in self.ALLOWED_IMAGE_TYPES:
            if mime_type in self.IMAGE_FORMAT_MAP:
                extensions.update(self.IMAGE_FORMAT_MAP[mime_type])
        return extensions

    def get_mime_type_from_extension(self, extension: str) -> str:
        """Get MIME type from file extension.

        Args:
            extension: File extension (e.g., '.jpg', 'jpg', or 'JPG')

        Returns:
            MIME type string or 'application/octet-stream' if unknown
        """
        # Normalize extension
        ext = extension.lower()
        if not ext.startswith('.'):
            ext = f'.{ext}'

        # Search for matching MIME type
        for mime_type, extensions in self.IMAGE_FORMAT_MAP.items():
            if ext in extensions:
                return mime_type

        return 'application/octet-stream'


def get_settings() -> Settings:
    """Get settings instance based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()

    # Load environment-specific .env file
    env_file = {
        "development": ".env",
        "staging": ".env.staging",
        "production": ".env.production",
        "testing": ".env.test"
    }.get(env, ".env")

    if Path(env_file).exists():
        os.environ["ENV_FILE"] = env_file

    return Settings()


# Global settings instance
settings = get_settings()
