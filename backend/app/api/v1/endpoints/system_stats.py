"""
System statistics endpoints.
Provides HTTP and SSE endpoints for real-time system monitoring.
"""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional
import logging
import asyncio
import json

from app.services.system_stats_service import system_stats_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/stats")
async def get_system_stats():
    """
    Get current system statistics snapshot.

    Returns:
        JSON containing CPU, memory, disk, GPU, network, and process stats
    """
    try:
        stats = system_stats_service.get_all_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve system statistics", "detail": str(e)}
        )


@router.get("/stats/stream")
async def stream_system_stats(request: Request, interval: Optional[float] = 1.0):
    """
    SSE endpoint for streaming system statistics.

    Query Parameters:
        interval: Update interval in seconds (default: 1.0)

    Returns:
        Server-Sent Events stream containing system stats at the specified interval
    """
    async def event_generator():
        """Generate SSE events with system stats"""
        try:
            logger.info(f"SSE connected for system stats (interval: {interval}s)")

            # Stream stats to the SSE client
            async for stats in system_stats_service.stream_stats(interval_seconds=interval):
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected")
                    break

                try:
                    # Format as SSE event
                    data = json.dumps(stats)
                    yield f"data: {data}\n\n"
                except Exception as e:
                    logger.error(f"Error formatting stats for SSE: {e}", exc_info=True)
                    break

        except asyncio.CancelledError:
            logger.info("SSE stream cancelled (server shutdown or client disconnect)")
            raise
        except Exception as e:
            logger.error(f"SSE streaming error: {e}", exc_info=True)
        finally:
            logger.info("SSE stream ended for system stats")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.get("/stats/cpu")
async def get_cpu_stats():
    """Get CPU statistics only."""
    try:
        stats = system_stats_service.get_cpu_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting CPU stats: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve CPU statistics", "detail": str(e)}
        )


@router.get("/stats/memory")
async def get_memory_stats():
    """Get memory statistics only."""
    try:
        stats = system_stats_service.get_memory_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve memory statistics", "detail": str(e)}
        )


@router.get("/stats/gpu")
async def get_gpu_stats():
    """Get GPU statistics only (if available)."""
    try:
        stats = system_stats_service.get_gpu_stats()
        if stats is None:
            return JSONResponse(content={"available": False, "message": "No GPU detected"})
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting GPU stats: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve GPU statistics", "detail": str(e)}
        )


@router.get("/stats/disk")
async def get_disk_stats():
    """Get disk statistics only."""
    try:
        stats = system_stats_service.get_disk_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting disk stats: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve disk statistics", "detail": str(e)}
        )


@router.get("/stats/network")
async def get_network_stats():
    """Get network statistics only."""
    try:
        stats = system_stats_service.get_network_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting network stats: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve network statistics", "detail": str(e)}
        )
