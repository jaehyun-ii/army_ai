"""
Shared SSE (Server-Sent Events) support for real-time logging.
Extracted from attack_support.py for general use across services.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, AsyncGenerator


class SSELogger:
    """Helper to mirror logs to both standard logging and SSE streams."""

    def __init__(
        self,
        base_logger: logging.Logger,
        sse_manager: Optional[Any],
        session_id: Optional[str],
    ) -> None:
        self._logger = base_logger
        self._sse_manager = sse_manager
        self._session_id = session_id

    async def _dispatch(self, log_type: str, message: str, **extra_data: Any) -> None:
        """Send a log message to SSE subscribers and the provided logger."""
        if self._sse_manager and self._session_id:
            payload = {
                "type": log_type,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            }
            if extra_data:
                payload.update(extra_data)

            self._logger.debug(f"[SSE] Sending to session {self._session_id}: {log_type} - {message}")
            await self._sse_manager.send_event(self._session_id, payload)
        else:
            self._logger.debug(f"[SSE] Not configured - manager: {self._sse_manager is not None}, session: {self._session_id}")

        log_method = getattr(self._logger, self._log_level(log_type), self._logger.info)
        log_method(message)

    @staticmethod
    def _log_level(log_type: str) -> str:
        """Map SSE log types to logging module level names."""
        return {
            "error": "error",
            "warning": "warning",
            "success": "info",
            "status": "info",
            "progress": "info",
            "info": "info",
        }.get(log_type, "info")

    async def info(self, message: str, **extra_data: Any) -> None:
        await self._dispatch("info", message, **extra_data)

    async def status(self, message: str, **extra_data: Any) -> None:
        await self._dispatch("status", message, **extra_data)

    async def progress(self, message: str, **extra_data: Any) -> None:
        await self._dispatch("progress", message, **extra_data)

    async def success(self, message: str, **extra_data: Any) -> None:
        await self._dispatch("success", message, **extra_data)

    async def warning(self, message: str, **extra_data: Any) -> None:
        await self._dispatch("warning", message, **extra_data)

    async def error(self, message: str, **extra_data: Any) -> None:
        await self._dispatch("error", message, **extra_data)


class SSEManager:
    """Shared SSE session manager for streaming outputs."""

    def __init__(self) -> None:
        self._event_queues: Dict[str, asyncio.Queue] = {}

    def create_session(self, session_id: str) -> None:
        """Create or reset an SSE session queue."""
        self._event_queues[session_id] = asyncio.Queue()

    def remove_session(self, session_id: str) -> None:
        """Remove SSE session."""
        self._event_queues.pop(session_id, None)

    async def send_event(self, session_id: str, message: Dict[str, Any]) -> None:
        """Enqueue event for a given session."""
        if session_id in self._event_queues:
            await self._event_queues[session_id].put(message)
            logger = logging.getLogger(__name__)
            logger.debug(f"[SSEManager] Queued message for session {session_id}: {message.get('type', 'unknown')}")
        else:
            logger = logging.getLogger(__name__)
            logger.warning(f"[SSEManager] Session {session_id} not found in active sessions: {list(self._event_queues.keys())}")

    async def event_stream(self, session_id: str) -> AsyncGenerator[str, None]:
        """Async generator yielding SSE formatted messages.

        Stream ends when:
        - A message with type='complete' or type='error' is sent
        - Client disconnects (asyncio.CancelledError)
        """
        if session_id not in self._event_queues:
            return

        queue = self._event_queues[session_id]
        try:
            while True:
                message = await queue.get()
                yield f"data: {json.dumps(message)}\n\n"

                if message.get("type") in ("complete", "error"):
                    break
        except asyncio.CancelledError:
            pass
        finally:
            self.remove_session(session_id)
