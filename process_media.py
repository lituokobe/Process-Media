# The service that engage with vector database
import asyncio
import json
from datetime import datetime
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from config.path_config import INPUT_DATA_PATH
from config.schema_config import DataPathRequest
from functionals.logger import process_media_logger
from functionals.milvus import initialize_milvus_async
from functionals.stream_manager import StreamManager

_process_lock = asyncio.Lock() # Lock the process and only process one request per time

app = FastAPI(title="Process Media Service")

def _read_json_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

async def _load_org_list(data_path: str|None = None, timeout: int = 30) -> list:
    if data_path:
        try:
            process_media_logger.info(f"📥 从远程获取数据: {data_path}")
            response = await asyncio.to_thread(
                lambda: requests.get(data_path, timeout=timeout)
            )
            response.raise_for_status()  # Fails fast on 4xx/5xx
            org_list = response.json()
            process_media_logger.info(f"✅ 成功获取 {len(org_list) if isinstance(org_list, list) else '未知'} 条组织数据")
        except Exception as e:
            e_m = f"❌ 从{data_path}获取数据出错: {e}"
            process_media_logger.error(e_m)
            raise HTTPException(status_code=502, detail=e_m) from e
    else:
        try:
            org_list = await asyncio.to_thread(_read_json_file, INPUT_DATA_PATH)
        except Exception as e:
            e_m = f"❌ 从{INPUT_DATA_PATH}获取数据出错: {e}"
            process_media_logger.error(e_m)
            raise HTTPException(status_code=500, detail=e_m) from e

    if not isinstance(org_list, list):
        e_m = "❌ 输入数据必须为数组/列表 (JSON array)"
        process_media_logger.error(e_m)
        raise HTTPException(status_code=400, detail=e_m)

    return org_list

async def _process_orgs_with_streaming(data_path: str | None, stream: StreamManager):
    """Internal processing function that streams results"""
    start_time = datetime.now()
    process_media_logger.info("♾️ 开始流式处理全量数据")

    org_list = await _load_org_list(data_path)

    total = len(org_list)
    progress_count = 0

    for idx, org in enumerate(org_list):
        org_id = org.get("org_id")

        # Send progress before processing each org
        await stream.send_progress(idx, total, org_id)

        if not isinstance(org, dict) or not isinstance(org_id, int):
            continue

        try:
            # Pass stream to the milvus initializer
            await initialize_milvus_async(org, stream=stream)
            progress_count += 1
        except Exception as e:
            process_media_logger.error(f"❌ 处理组织{org_id}出错: {e}")
            await stream.send("error", {
                "org_id": org_id,
                "message": str(e),
                "stage": "initialize_milvus"
            })

    # Build and send final summary
    duration_seconds = (datetime.now() - start_time).total_seconds()
    summary = {
        "status": "completed",
        "total_requested": total,
        "successfully_processed": progress_count,
        "failed_or_skipped": total - progress_count,
        "start_time": start_time.isoformat(),
        "duration_seconds": int(duration_seconds),
        "duration_hours": round(duration_seconds / 3600, 2)
    }

    await stream.send_complete(summary)
    process_media_logger.info(f"♾️ 流式处理结束: {progress_count}/{total}")

@app.post("/stream_process")
async def stream_process(request: DataPathRequest):
    """
    Stream process media data for all organizations.
    Only one request can be processed at a time.
    Returns 429 if service is busy.

    Stream progressive results via SSE.
    Returns entry payloads as they're generated + final summary.
    """
    # 🚫 Non-blocking check: reject immediately if already processing
    if _process_lock.locked():
        process_media_logger.warning("⚠️ 服务正忙，拒绝新请求")
        raise HTTPException(
            status_code=429,
            detail="服务正处理其他请求，请稍后重试"
        )

    # 🔐 Acquire lock using context manager (auto-releases on exit)
    async with _process_lock:
        stream = StreamManager()

        async def event_generator():
            try:
                # Yield initial connection event
                await stream.send("connected", {"message": "处理开始"})
                task = asyncio.create_task(_process_orgs_with_streaming(request.data_path, stream))
                async for event in stream.stream():
                    yield event
                await task

            except asyncio.CancelledError:
                # Client disconnected
                stream.disconnect()
                process_media_logger.info("🔌 客户端断开连接，停止流式输出")
                raise
            except Exception as e:
                # Send error event before closing
                if not getattr(stream, '_completed', False):
                    await stream.send("error", {"message": str(e), "type": type(e).__name__})
                raise

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable Nginx buffering
            }
        )
    
@app.post("/process")
async def process(request: DataPathRequest):
    """
    Process media data for all organizations.
    Only one request can be processed at a time.
    Returns 429 if service is busy.
    """
    # 🚫 Non-blocking check: reject immediately if already processing
    if _process_lock.locked():
        process_media_logger.warning("⚠️ 服务正忙，拒绝新请求")
        raise HTTPException(
            status_code=429,
            detail="服务正处理其他请求，请稍后重试"
        )

    # 🔐 Acquire lock using context manager (auto-releases on exit)
    async with _process_lock:
        try:
            start_time = datetime.now()
            process_media_logger.info(f"♾️ 开始处理全量数据")
            data_path = request.data_path
            org_list = await _load_org_list(data_path)

            # -------- Process each organization --------
            progress_count = 0
            total = len(org_list)
            for org in org_list:
                # -------- 1. Validate the org data --------
                if not isinstance(org, dict):
                    e_m = "❌ 组织数据需为字典"
                    process_media_logger.error(e_m)
                    continue

                org_id = org.get("org_id")
                if not isinstance(org_id, int):
                    e_m = "❌ 组织org_id缺少或有误"
                    process_media_logger.error(e_m)
                    continue

                try:
                    await initialize_milvus_async(org)
                    progress_count+=1
                except Exception as e:
                    e_m = f"❌ 处理组织{org_id}出错: {e}"
                    process_media_logger.error(e_m)

            duration_seconds = (datetime.now() - start_time).total_seconds()
            duration_hours = round(duration_seconds / 3600, 2)

            process_media_logger.info(f"♾️ 全量数据处理结束, 处理组织: {progress_count}/{total}, 耗时{duration_hours}小时")
            return {
                "status": "completed",
                "total_requested": total,
                "successfully_processed": progress_count,
                "failed_or_skipped": total - progress_count,
                "start_time":start_time.isoformat(),
                "duration_seconds": int(duration_seconds),
                "duration_hours": duration_hours
            }
        except HTTPException:
            # Re-raise FastAPI HTTP exceptions as-is
            raise
        except Exception as e:
            # Catch unexpected errors, log, and return 500
            process_media_logger.exception(f"❌ 处理数据错误: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"内部处理错误: {str(e)}"
            ) from e
        
@app.get("/health")
async def health_check():
    """Docker & monitoring health endpoint"""
    return {
        "status": "healthy",
        "service": "process-media",
        "timestamp": datetime.now().isoformat()
    }