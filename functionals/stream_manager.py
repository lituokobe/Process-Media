import asyncio
import json
from datetime import datetime
from typing import Any, AsyncGenerator

class StreamManager:
    """Manages streaming results for a single request"""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._client_disconnected = False

    async def send(self, event_type: str, data: Any):
        """Send an event to the stream"""
        if self._client_disconnected:
            return
        payload = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        await self._queue.put(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")

    async def send_entry(self, payload: dict):
        """Convenience method for entry payloads"""
        await self.send("entry", payload)

    async def send_progress(self, current: int, total: int, org_id: int = None):
        """Send progress update"""
        await self.send("progress", {
            "current": current,
            "total": total,
            "org_id": org_id,
            "percent": round(current / total * 100, 2) if total > 0 else 0
        })

    async def send_complete(self, summary: dict):
        """Send final completion event"""
        await self.send("complete", summary)
        await self._queue.put(None)  # Sentinel to end stream

    async def stream(self) -> AsyncGenerator[str, None]:
        """Async generator for StreamingResponse"""
        try:
            while True:
                item = await self._queue.get()
                if item is None:  # End of stream
                    break
                try:
                    yield item
                except (ConnectionError, BrokenPipeError):
                    self._client_disconnected = True
                    raise
        except asyncio.CancelledError:
            self._client_disconnected = True
            raise

    def disconnect(self):
        """Mark client as disconnected"""
        self._client_disconnected = True