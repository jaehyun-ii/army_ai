"""
System statistics monitoring service.
Provides real-time CPU, GPU, memory, and disk usage stats.
"""
import psutil
import time
from typing import Dict, AsyncGenerator, Optional, Any
import asyncio
import math


class SystemStatsService:
    """Service for monitoring system resource usage."""

    def __init__(self):
        self.gpu_available = self._check_gpu_available()

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
        """Check if GPU monitoring is available."""
        try:
            import GPUtil
            return True
        except ImportError:
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

    def get_gpu_stats(self) -> Optional[Dict]:
        """Get GPU usage statistics (if available)."""
        if not self.gpu_available:
            return None

        try:
            import GPUtil
            gpus = GPUtil.getGPUs()

            if not gpus:
                return None

            gpu_stats = []
            for gpu in gpus:
                gpu_stats.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "load_percent": round(gpu.load * 100, 2),
                    "memory_total_mb": round(gpu.memoryTotal, 2),
                    "memory_used_mb": round(gpu.memoryUsed, 2),
                    "memory_free_mb": round(gpu.memoryFree, 2),
                    "memory_percent": round((gpu.memoryUsed / gpu.memoryTotal) * 100, 2),
                    "temperature_c": round(gpu.temperature, 2),
                })

            return {
                "available": True,
                "count": len(gpus),
                "gpus": gpu_stats,
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
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
