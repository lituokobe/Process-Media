import asyncio
import json
import os
import subprocess
import time
from enum import Enum
import requests
from config.constant_config import COLLECTION_NAME_PREFIX
from config.path_config import MULTIMODEL_SERVICE_URL, BGM_SERVICE_URL, OUTPUT_DATA_PATH, OUTPUT_REQUEST_URL
from functionals.logger import process_media_logger
from functionals.stream_manager import StreamManager

DOCKER_CMD = os.getenv("DOCKER_CMD", "docker")

def user_id_to_collection_name(user_id:str|int|None)->str:
    return COLLECTION_NAME_PREFIX + str(user_id)

class ModelService(Enum):
    """Supported model services."""
    # Enum is less flexible, more safe. Designed to restrict valid values and prevent runtime bugs.
    MULTIMODAL_SERVICE = "multimodal_service"   # Qwen3.5-9B
    BGM_SERVICE = "bgm_service"   # Qwen-Audio-Chat

class ModelServiceOrchestrator:
    """
    Manages lifecycle of mutually-exclusive Docker-based model services.
    Ensures only one service runs at a time, with blocking health checks.
    """

    # Service configuration: name → settings
    SERVICE_CONFIG = {
        ModelService.MULTIMODAL_SERVICE: {
            "container_name": "multimodal_summarization",
            "health_url": MULTIMODEL_SERVICE_URL + "/health",
            "check_start_min": 6,  # Start probing after 6 min
            "check_end_min": 10,  # Fail if not healthy by 8 min
        },
        ModelService.BGM_SERVICE: {
            "container_name": "bgm_summarization",
            "health_url": BGM_SERVICE_URL + "/health",
            "check_start_min": 4,  # Start probing after 4 min
            "check_end_min": 8,  # Fail if not healthy by 6 min
        }
    }

    def __init__(self, poll_interval_sec: int = 25, docker_timeout_sec: int = 30):
        """
        Initialize the orchestrator.

        Args:
            poll_interval_sec: How often to poll health endpoint during wait window
            docker_timeout_sec: Timeout for docker stop command (grace period)
        """
        self._poll_interval = poll_interval_sec
        self._docker_timeout = docker_timeout_sec
        self._current_service: ModelService|None = None  # Track active service

    def switch_to(self, service: ModelService) -> None:
        """
        Switch to the specified model service (blocking until healthy).

        Flow:
        1. If target service already running → return immediately
        2. Stop the OTHER service (if running)
        3. Start the target service
        4. Block until health check passes (with service-specific timing)

        Args:
            service: ModelService.MULTIMODAL_SERVICE or ModelService.BGM_SERVICE

        Raises:
            RuntimeError: If service fails to become healthy within its timeout window
            subprocess.CalledProcessError: If docker command fails unexpectedly
        """
        config = self.SERVICE_CONFIG[service]
        other_service = ModelService.MULTIMODAL_SERVICE if service == ModelService.BGM_SERVICE else ModelService.BGM_SERVICE
        other_config = self.SERVICE_CONFIG[other_service]

        # Query ACTUAL Docker state (source of truth)
        target_status = self._get_container_status(config["container_name"])
        other_status = self._get_container_status(other_config["container_name"])

        # Stop the other container if it's running
        if other_status == "running":
            process_media_logger.info(f"ℹ️ 停止{other_service.value}服务 ({other_config['container_name']}容器)...")
            self._stop_container(other_config["container_name"])
            time.sleep(2)  # Allow NVIDIA driver to release VRAM

        if target_status != "running":
            process_media_logger.info(f"🚀 开启{service.value}服务 ({config['container_name']}容器)...")
            self._start_container(config["container_name"])

            self._wait_for_healthy(
                service_name=service.value,
                health_url=config["health_url"],
                check_start_sec=config["check_start_min"] * 60,
                check_end_sec=config["check_end_min"] * 60,
            )

        self._current_service = service
        if self._quick_health_check(config["health_url"]):
            process_media_logger.info(f"✅ {service.value}服务正在运行并且健康")
        else:
            process_media_logger.warning(f"⚠️ {service.value}服务不健康")
        return

    @staticmethod
    def _get_container_status(name: str) -> str:
        """Return actual Docker container status: running, exited, created, or unknown."""
        try:
            result = subprocess.run(
                [DOCKER_CMD, "inspect", "-f", "{{.State.Status}}", name],
                capture_output=True, text=True, check=True, encoding="utf-8", timeout=10
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"

    def _stop_container(self, container_name: str) -> None:
        """Stop a docker container gracefully. Safe to call if already stopped."""
        try:
            # Check current status
            result = subprocess.run(
                [DOCKER_CMD, "inspect", "-f", "{{.State.Status}}", container_name],
                capture_output=True, text=True, encoding="utf-8", check=True, timeout=10
            )
            if result.stdout.strip() != "running":
                process_media_logger.info(f"{container_name}容器没有在运行 (status: {result.stdout.strip()}), 跳过关闭")
                return

            # Stop with graceful timeout
            subprocess.run(
                [DOCKER_CMD, "stop", "--time", str(self._docker_timeout), container_name],
                capture_output=True, text=True, encoding="utf-8", check=True, timeout=self._docker_timeout + 10
            )
            process_media_logger.info(f"{container_name}容器成功关闭")

        except subprocess.CalledProcessError as e:
            if "No such container" in e.stderr:
                process_media_logger.error(f"无法找到{container_name}容器, 跳过关闭")
            else:
                process_media_logger.error(f"无法关闭{container_name}容器: {e.stderr.strip()}")
                raise

    @staticmethod
    def _start_container(container_name: str) -> None:
        """Start a docker container."""
        subprocess.run(
            [DOCKER_CMD, "start", container_name],
            capture_output=True, text=True, encoding="utf-8", check=True, timeout=30
        )
        process_media_logger.info(f"{container_name}容器开始启动")

    @staticmethod
    def _quick_health_check(health_url: str) -> bool:
        """Non-blocking probe. Returns True if healthy, False otherwise."""
        try:
            resp = requests.get(health_url, timeout=5)
            return resp.status_code == 200 and resp.json().get("status") == "healthy"
        except requests.RequestException:
            raise False

    def _wait_for_healthy(
            self,
            service_name: str,
            health_url: str,
            check_start_sec: int,
            check_end_sec: int,
    ) -> None:
        """
        Block until health check returns {"status": "healthy"}.

        - Silent wait for check_start_sec (no probes)
        - Then poll every self._poll_interval seconds until check_end_sec
        - Return immediately on first healthy response
        - Raise RuntimeError if timeout reached without success
        """
        process_media_logger.info(
            f"⏳ {service_name}健康检查: "
            f"{check_start_sec // 60}分钟后开始检查, 每{self._poll_interval}秒一次, 直到{check_end_sec // 60}分钟后"
        )

        # Phase 1: Silent initial wait (no network probes)
        time.sleep(check_start_sec)

        # Phase 2: Polling window
        start_time = time.time()
        end_time = start_time + (check_end_sec - check_start_sec)

        while time.time() < end_time:
            try:
                resp = requests.get(health_url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "healthy":
                        process_media_logger.info(f"💚 {service_name}服务健康检查通过: {data}")
                        return
                    process_media_logger.warning(f"⚠️ {service_name}服务不健康: {data}")
                else:
                    process_media_logger.info(f"❌ 健康检查返回 HTTP {resp.status_code}")
            except requests.RequestException as e:
                process_media_logger.error(f"❌ 健康检查请求失败: {e}")

            time.sleep(self._poll_interval)

        # Timeout without success
        raise RuntimeError(
            f"❌ {service_name}服务在{check_end_sec // 60}分钟内无法健康启动, "
            f"检查链接: {health_url}"
        )

    def get_active_service(self) -> ModelService|None:
        """Return the currently active service, or None if none."""
        return self._current_service

    def shutdown_all(self) -> None:
        """Stop all managed containers. Useful for cleanup on application exit."""
        for config in self.SERVICE_CONFIG.values():
            self._stop_container(config["container_name"])
        self._current_service = None
        process_media_logger.info("🏁 两个大模型服务均已关闭")

def _write_payload_sync(payload: dict):
    with open(OUTPUT_DATA_PATH, "a", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")

async def _safe_post(url: str, payload: dict, service_name: str):
    """A simple, reusable helper to post JSON data safely without blocking the async loop."""
    try:
        response = await asyncio.to_thread(
            lambda: requests.post(url, json=payload, timeout=10, verify=True)
        )
        response.raise_for_status()
        process_media_logger.info(f"✅ {service_name} POST请求成功: {response.status_code}")
    except Exception as e:
        process_media_logger.error(f"❌ {service_name} POST请求失败: {e}")

async def stream_post_payload(payload: dict, stream: StreamManager = None):
    # stream the data
    if stream:
        await stream.send_entry(payload)

    # post the data
    await asyncio.to_thread(_write_payload_sync, payload)

    # 3. Post to general OUTPUT_REQUEST_URL, database in this case. Add more with _safe_post if needed
    if isinstance(OUTPUT_REQUEST_URL, str):
        await _safe_post(OUTPUT_REQUEST_URL, payload, "Output API")


if __name__ == "__main__":
    MSO = ModelServiceOrchestrator()
    MSO.switch_to(ModelService.BGM_SERVICE)
    # MSO.switch_to(ModelService.MULTIMODAL_SERVICE)