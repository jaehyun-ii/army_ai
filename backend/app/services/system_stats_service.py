"""
System statistics monitoring service.
Provides real-time CPU, GPU, memory, and disk usage stats.
"""
import psutil
import time
import subprocess
import xml.etree.ElementTree as ET
from typing import Dict, AsyncGenerator, Optional, Any
import asyncio
import math
import logging

logger = logging.getLogger(__name__)


class SystemStatsService:
    """Service for monitoring system resource usage."""

    def __init__(self):
        self.gpu_available = self._check_gpu_available()

    @staticmethod
    def _parse_float_safe(text: str, default: float = 0.0) -> float:
        """Safely parse float from text, handling 'N/A' and other invalid values."""
        if not text:
            return default

        text = text.strip()

        # Handle 'N/A' or similar
        if text.upper() in ('N/A', 'NA', 'NONE', '-'):
            return default

        try:
            # Remove common units and parse
            text = text.split()[0]  # Take first word (removes units like 'MiB', '%', 'C')
            return float(text)
        except (ValueError, IndexError):
            return default

    @staticmethod
    def _sanitize_value(value: Any) -> Any:
        """Sanitize values to ensure JSON serialization compatibility."""
        if value is None:
            return None
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
        if isinstance(value, dict):
            return {k: SystemStatsService._sanitize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [SystemStatsService._sanitize_value(v) for v in value]
        return value

    def _check_gpu_available(self) -> bool:
        """Check if GPU monitoring is available (using pynvml)."""
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"pynvml check: found {device_count} GPU(s)")
            return device_count > 0
        except ImportError:
            logger.warning("pynvml (nvidia-ml-py) not installed")
            return False
        except Exception as e:
            logger.warning(f"pynvml check failed: {e}")
            return False

    def get_cpu_stats(self) -> Dict:
        """Get CPU usage statistics."""
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)

        # Try to get CPU frequency (may fail on some systems like macOS)
        try:
            cpu_freq = psutil.cpu_freq()
        except (OSError, FileNotFoundError):
            cpu_freq = None

        return {
            "usage_percent": round(sum(cpu_percent) / len(cpu_percent), 2),
            "usage_per_core": [round(p, 2) for p in cpu_percent],
            "core_count": psutil.cpu_count(logical=False),
            "thread_count": psutil.cpu_count(logical=True),
            "frequency_mhz": round(cpu_freq.current, 2) if cpu_freq else None,
            "frequency_max_mhz": round(cpu_freq.max, 2) if cpu_freq else None,
        }

    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return {
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "used_gb": round(mem.used / (1024**3), 2),
            "percent": round(mem.percent, 2),
            "swap_total_gb": round(swap.total / (1024**3), 2),
            "swap_used_gb": round(swap.used / (1024**3), 2),
            "swap_percent": round(swap.percent, 2),
        }

    def get_disk_stats(self) -> Dict:
        """Get disk usage statistics."""
        disk = psutil.disk_usage('/')
        io = psutil.disk_io_counters()

        return {
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent": round(disk.percent, 2),
            "read_mb": round(io.read_bytes / (1024**2), 2) if io else None,
            "write_mb": round(io.write_bytes / (1024**2), 2) if io else None,
        }

    def _get_gpu_stats_pynvml(self) -> Optional[Dict]:
        """Get GPU stats using pynvml (NVIDIA Management Library)."""
        try:
            import pynvml

            # Initialize NVML if not already done
            try:
                pynvml.nvmlInit()
            except pynvml.NVMLError:
                pass  # Already initialized

            device_count = pynvml.nvmlDeviceGetCount()

            if device_count == 0:
                logger.warning("pynvml: No GPUs found")
                return None

            gpu_stats = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Get GPU name
                try:
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                except pynvml.NVMLError:
                    name = f"GPU {i}"

                # Get memory info - try multiple methods
                # Note: Some GPUs (e.g., shared memory vGPU) don't support memory queries
                memory_total_mb = None
                memory_used_mb = None
                memory_free_mb = None
                memory_percent = None

                # Method 1: Try standard memory info
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_total_mb = mem_info.total / (1024 ** 2)
                    memory_used_mb = mem_info.used / (1024 ** 2)
                    memory_free_mb = mem_info.free / (1024 ** 2)
                    memory_percent = (mem_info.used / mem_info.total * 100) if mem_info.total > 0 else 0
                    logger.debug(f"GPU {i}: Got memory info via nvmlDeviceGetMemoryInfo")
                except pynvml.NVMLError as e:
                    logger.debug(f"GPU {i}: nvmlDeviceGetMemoryInfo not supported (shared memory GPU): {e}")

                    # Method 2: Try BAR1 memory (for virtual GPUs)
                    try:
                        bar1_mem = pynvml.nvmlDeviceGetBAR1MemoryInfo(handle)
                        memory_total_mb = bar1_mem.bar1Total / (1024 ** 2)
                        memory_used_mb = bar1_mem.bar1Used / (1024 ** 2)
                        memory_free_mb = bar1_mem.bar1Free / (1024 ** 2)
                        memory_percent = (bar1_mem.bar1Used / bar1_mem.bar1Total * 100) if bar1_mem.bar1Total > 0 else 0
                        logger.debug(f"GPU {i}: Got memory info via BAR1")
                    except pynvml.NVMLError as e2:
                        logger.debug(f"GPU {i}: Memory queries not supported on this GPU (shared memory architecture)")

                # Get utilization (GPU load)
                load_percent = None
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    load_percent = float(util.gpu)
                    logger.debug(f"GPU {i}: Load = {load_percent}%")
                except pynvml.NVMLError as e:
                    logger.debug(f"GPU {i}: Utilization not supported: {e}")

                # Get temperature
                temp = None
                try:
                    temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                    logger.debug(f"GPU {i}: Temperature = {temp}°C")
                except pynvml.NVMLError as e:
                    logger.debug(f"GPU {i}: Temperature not supported: {e}")

                # Get power usage
                power_watts = None
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    power_watts = power_mw / 1000.0  # Convert milliwatts to watts
                    logger.debug(f"GPU {i}: Power = {power_watts}W")
                except pynvml.NVMLError as e:
                    logger.debug(f"GPU {i}: Power usage not supported: {e}")

                mem_str = f"{memory_used_mb:.0f}/{memory_total_mb:.0f}MB" if memory_total_mb else "N/A"
                logger.info(
                    f"GPU {i} ({name}): "
                    f"Load={load_percent}%, "
                    f"Mem={mem_str}, "
                    f"Temp={temp}°C, "
                    f"Power={power_watts}W"
                )

                gpu_stats.append({
                    "id": i,
                    "name": name,
                    "load_percent": round(load_percent, 2) if load_percent is not None else None,
                    "memory_total_mb": round(memory_total_mb, 2) if memory_total_mb is not None else None,
                    "memory_used_mb": round(memory_used_mb, 2) if memory_used_mb is not None else None,
                    "memory_free_mb": round(memory_free_mb, 2) if memory_free_mb is not None else None,
                    "memory_percent": round(memory_percent, 2) if memory_percent is not None else None,
                    "temperature_c": round(temp, 2) if temp is not None else None,
                    "power_watts": round(power_watts, 2) if power_watts is not None else None,
                })

            logger.debug(f"pynvml: {len(gpu_stats)} GPU(s) stats collected")
            return {
                "available": True,
                "count": len(gpu_stats),
                "gpus": gpu_stats,
            }
        except ImportError:
            logger.error("pynvml is not installed")
            return None
        except Exception as e:
            logger.error(f"pynvml failed: {e}")
            logger.debug(f"pynvml error details", exc_info=True)
            return None

    def _get_gpu_stats_nvidia_smi_OLD(self) -> Optional[Dict]:
        """Get GPU stats using nvidia-smi directly (most reliable method)."""
        try:
            # Run nvidia-smi with XML output
            result = subprocess.run(
                ['nvidia-smi', '-q', '-x'],
                capture_output=True,
                timeout=5,
                check=True,
                text=True
            )

            if result.returncode != 0:
                logger.warning(f"nvidia-smi returned non-zero exit code: {result.returncode}")
                return None

            # Parse XML
            root = ET.fromstring(result.stdout)
            gpu_stats = []

            logger.debug(f"nvidia-smi XML parsed, found {len(root.findall('gpu'))} GPUs")

            for idx, gpu in enumerate(root.findall('gpu')):
                # Get name
                name_elem = gpu.find('product_name')
                name = name_elem.text if name_elem is not None else f"GPU {idx}"

                # Get utilization
                util_elem = gpu.find('.//utilization/gpu_util')
                load_percent = self._parse_float_safe(util_elem.text if util_elem is not None else None, 0.0)

                # Get memory - try multiple possible paths
                memory_total_mb = 0.0
                memory_used_mb = 0.0
                memory_free_mb = 0.0

                # Try fb_memory_usage first
                mem_total_elem = gpu.find('.//fb_memory_usage/total')
                mem_used_elem = gpu.find('.//fb_memory_usage/used')
                mem_free_elem = gpu.find('.//fb_memory_usage/free')

                if mem_total_elem is not None:
                    memory_total_mb = self._parse_float_safe(mem_total_elem.text, 0.0)
                    logger.debug(f"GPU {idx} fb_memory total: {mem_total_elem.text} -> {memory_total_mb}")

                if mem_used_elem is not None:
                    memory_used_mb = self._parse_float_safe(mem_used_elem.text, 0.0)
                    logger.debug(f"GPU {idx} fb_memory used: {mem_used_elem.text} -> {memory_used_mb}")

                if mem_free_elem is not None:
                    memory_free_mb = self._parse_float_safe(mem_free_elem.text, 0.0)

                # If fb_memory_usage failed, try bar1_memory_usage
                if memory_total_mb == 0.0:
                    mem_total_elem = gpu.find('.//bar1_memory_usage/total')
                    mem_used_elem = gpu.find('.//bar1_memory_usage/used')
                    mem_free_elem = gpu.find('.//bar1_memory_usage/free')

                    if mem_total_elem is not None:
                        memory_total_mb = self._parse_float_safe(mem_total_elem.text, 0.0)
                        logger.debug(f"GPU {idx} bar1_memory total: {mem_total_elem.text} -> {memory_total_mb}")

                    if mem_used_elem is not None:
                        memory_used_mb = self._parse_float_safe(mem_used_elem.text, 0.0)

                    if mem_free_elem is not None:
                        memory_free_mb = self._parse_float_safe(mem_free_elem.text, 0.0)

                memory_percent = (memory_used_mb / memory_total_mb * 100) if memory_total_mb > 0 else 0

                # Get temperature
                temp_elem = gpu.find('.//temperature/gpu_temp')
                temperature_c = self._parse_float_safe(temp_elem.text if temp_elem is not None else None, 0.0)

                logger.debug(f"GPU {idx} final stats - Total: {memory_total_mb}MB, Used: {memory_used_mb}MB, Temp: {temperature_c}C")

                gpu_stats.append({
                    "id": idx,
                    "name": name,
                    "load_percent": round(load_percent, 2),
                    "memory_total_mb": round(memory_total_mb, 2),
                    "memory_used_mb": round(memory_used_mb, 2),
                    "memory_free_mb": round(memory_free_mb, 2),
                    "memory_percent": round(memory_percent, 2),
                    "temperature_c": round(temperature_c, 2),
                })

            if not gpu_stats:
                return None

            return {
                "available": True,
                "count": len(gpu_stats),
                "gpus": gpu_stats,
            }

        except Exception as e:
            logger.warning(f"nvidia-smi failed: {e}")
            logger.debug(f"nvidia-smi error details", exc_info=True)
            return None

    def _get_gpu_stats_gputil(self) -> Optional[Dict]:
        """Get GPU stats using GPUtil."""
        try:
            import GPUtil
            import math
            gpus = GPUtil.getGPUs()

            if not gpus:
                logger.warning("GPUtil returned no GPUs")
                return None

            gpu_stats = []
            for gpu in gpus:
                # GPUtil returns memory in MB, load as 0-1 scale
                # Handle NaN and None values
                def safe_value(val, default=0.0):
                    """Convert value to float, handling NaN, None, and infinity."""
                    if val is None:
                        return default
                    if isinstance(val, (int, float)):
                        if math.isnan(val) or math.isinf(val):
                            return default
                        return float(val)
                    return default

                load_percent = round(safe_value(gpu.load, 0.0) * 100, 2)
                memory_total = round(safe_value(gpu.memoryTotal, 0.0), 2)
                memory_used = round(safe_value(gpu.memoryUsed, 0.0), 2)
                memory_free = round(safe_value(gpu.memoryFree, 0.0), 2)
                temperature = round(safe_value(gpu.temperature, 0.0), 2)

                logger.debug(
                    f"GPU {gpu.id} ({gpu.name}): "
                    f"Load={load_percent}%, "
                    f"Mem={memory_used}/{memory_total}MB, "
                    f"Temp={temperature}°C"
                )

                gpu_stats.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "load_percent": load_percent,
                    "memory_total_mb": memory_total,
                    "memory_used_mb": memory_used,
                    "memory_free_mb": memory_free,
                    "memory_percent": round((memory_used / memory_total * 100) if memory_total > 0 else 0, 2),
                    "temperature_c": temperature,
                })

            logger.debug(f"GPUtil: {len(gpu_stats)} GPU(s) stats collected")
            return {
                "available": True,
                "count": len(gpus),
                "gpus": gpu_stats,
            }
        except ImportError:
            logger.error("GPUtil is not installed")
            return None
        except Exception as e:
            logger.error(f"GPUtil failed: {e}")
            logger.debug(f"GPUtil error details", exc_info=True)
            return None

    def get_gpu_stats(self) -> Optional[Dict]:
        """Get GPU usage statistics using pynvml (NVIDIA Management Library)."""
        if not self.gpu_available:
            logger.debug("GPU not available, skipping GPU stats")
            return None

        # Use pynvml (NVIDIA Management Library)
        logger.debug("Getting GPU stats via pynvml...")
        result = self._get_gpu_stats_pynvml()
        if result is not None:
            logger.debug("Successfully got GPU stats from pynvml")
            return result

        # If failed
        logger.error("pynvml failed to get GPU stats")
        return {
            "available": False,
            "error": "Unable to get GPU stats from pynvml"
        }

    def get_network_stats(self) -> Dict:
        """Get network I/O statistics."""
        net = psutil.net_io_counters()

        return {
            "bytes_sent_mb": round(net.bytes_sent / (1024**2), 2),
            "bytes_recv_mb": round(net.bytes_recv / (1024**2), 2),
            "packets_sent": net.packets_sent,
            "packets_recv": net.packets_recv,
        }

    def get_process_stats(self) -> Dict:
        """Get current process statistics."""
        process = psutil.Process()

        with process.oneshot():
            mem_info = process.memory_info()
            cpu_percent = process.cpu_percent(interval=0.1)

            return {
                "pid": process.pid,
                "cpu_percent": round(cpu_percent, 2),
                "memory_mb": round(mem_info.rss / (1024**2), 2),
                "threads": process.num_threads(),
                "open_files": len(process.open_files()),
            }

    def get_all_stats(self) -> Dict:
        """Get all system statistics."""
        stats = {
            "timestamp": time.time(),
            "cpu": self.get_cpu_stats(),
            "memory": self.get_memory_stats(),
            "disk": self.get_disk_stats(),
            "gpu": self.get_gpu_stats(),
            "network": self.get_network_stats(),
            "process": self.get_process_stats(),
        }
        return self._sanitize_value(stats)

    async def stream_stats(
        self,
        interval_seconds: float = 1.0,
        max_samples: Optional[int] = None,
    ) -> AsyncGenerator[Dict, None]:
        """
        Stream system statistics at regular intervals.

        Args:
            interval_seconds: Time between samples (default: 1.0 seconds)
            max_samples: Maximum number of samples (None = infinite)

        Yields:
            Dict containing system statistics
        """
        sample_count = 0

        try:
            while True:
                # Check if we've reached max samples
                if max_samples is not None and sample_count >= max_samples:
                    break

                # Get stats
                stats = self.get_all_stats()
                stats["sample_number"] = sample_count + 1
                yield stats

                sample_count += 1

                # Wait for next interval - will raise CancelledError on shutdown
                await asyncio.sleep(interval_seconds)

        except asyncio.CancelledError:
            # Gracefully handle cancellation during shutdown
            raise


# Global instance
system_stats_service = SystemStatsService()
