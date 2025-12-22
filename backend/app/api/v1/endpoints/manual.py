"""
Manual download endpoint.
Provides user manual file download without database dependency.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import logging

from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/download")
async def download_manual():
    """
    Download user manual PDF file.

    This endpoint provides the user manual without requiring database lookup.
    The file is stored in storage/manual/manual.pdf (hardcoded path).

    **Returns:**
    PDF file download with Korean filename
    """
    # Hardcoded storage key (same pattern as patches, models, etc.)
    # DB에 없는 파일이므로 storage_key를 하드코딩
    storage_key = "manual/manual.pdf"

    # Construct full path (same pattern as other downloads)
    storage_root = Path(settings.STORAGE_ROOT)
    manual_path = storage_root / storage_key

    logger.info(f"Manual download requested. Storage key: {storage_key}")
    logger.info(f"Full path: {manual_path}")

    # Check file existence (same pattern as other downloads)
    if not manual_path.exists():
        logger.error(f"Manual file not found: {storage_key}")
        raise HTTPException(
            status_code=404,
            detail=f"Manual file not found: {storage_key}"
        )

    if not manual_path.is_file():
        logger.error(f"Manual path is not a file: {storage_key}")
        raise HTTPException(
            status_code=400,
            detail="Invalid manual file path"
        )

    # Return file with Korean filename and proper headers
    return FileResponse(
        path=str(manual_path),
        media_type="application/pdf",
        filename="AI_모델_신뢰성_검증_시스템_운용자_메뉴얼.pdf",
        headers={
            "Content-Disposition": "attachment; filename*=UTF-8''AI_%EB%AA%A8%EB%8D%B8_%EC%8B%A0%EB%A2%B0%EC%84%B1_%EA%B2%80%EC%A6%9D_%EC%8B%9C%EC%8A%A4%ED%85%9C_%EC%82%AC%EC%9A%A9%EC%9E%90_%EB%A9%94%EB%89%B4%EC%96%BC.pdf",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )
