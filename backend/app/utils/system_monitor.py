"""
System resource monitoring utility.
Monitors CPU, GPU, RAM, and Disk usage.
"""
import psutil
import platform
from typing import Dict, Any, List, Optional
from datetime import datetime


class SystemMonitor:
    """Monitor system resources (CPU, GPU, RAM, Disk)."""

    def __init__(self):
        """Initialize system monitor."""
        self.gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            return True
        except (ImportError, Exception):
            return False

    def get_cpu_metrics(self) -> Dict[str, Any]:
        """Get CPU metrics.

        Returns:
            Dictionary with CPU information
        """
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_freq = psutil.cpu_freq()

        return {
            "usage_percent": round(psutil.cpu_percent(interval=1), 2),
            "usage_per_core": [round(p, 2) for p in cpu_percent],
            "core_count": psutil.cpu_count(logical=False),
            "thread_count": psutil.cpu_count(logical=True),
            "frequency_mhz": {
                "current": round(cpu_freq.current, 2) if cpu_freq else None,
                "min": round(cpu_freq.min, 2) if cpu_freq else None,
                "max": round(cpu_freq.max, 2) if cpu_freq else None,
            },
            "load_average": {
                "1min": round(psutil.getloadavg()[0], 2),
                "5min": round(psutil.getloadavg()[1], 2),
                "15min": round(psutil.getloadavg()[2], 2),
            } if hasattr(psutil, 'getloadavg') else None,
        }

    def get_gpu_metrics(self) -> List[Dict[str, Any]]:
        """Get GPU metrics (supports up to 4 GPUs).

        Returns:
            List of GPU information dictionaries
        """
        if not self.gpu_available:
            return []

        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = min(pynvml.nvmlDeviceGetCount(), 4)  # Max 4 GPUs

            gpu_info = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Get GPU name
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')

                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                # Get utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                # Get temperature
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = None

                # Get power usage
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                except:
                    power_usage = None
                    power_limit = None

                gpu_info.append({
                    "index": i,
                    "name": name,
                    "memory": {
                        "total_mb": round(mem_info.total / 1024 / 1024, 2),
                        "used_mb": round(mem_info.used / 1024 / 1024, 2),
                        "free_mb": round(mem_info.free / 1024 / 1024, 2),
                        "usage_percent": round((mem_info.used / mem_info.total) * 100, 2),
                    },
                    "utilization": {
                        "gpu_percent": utilization.gpu,
                        "memory_percent": utilization.memory,
                    },
                    "temperature_celsius": temperature,
                    "power": {
                        "usage_watts": round(power_usage, 2) if power_usage else None,
                        "limit_watts": round(power_limit, 2) if power_limit else None,
                    },
                })

            pynvml.nvmlShutdown()
            return gpu_info

        except Exception as e:
            return []

    def get_memory_metrics(self) -> Dict[str, Any]:
        """Get RAM metrics.

        Returns:
            Dictionary with memory information
        """
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return {
            "total_mb": round(mem.total / 1024 / 1024, 2),
            "available_mb": round(mem.available / 1024 / 1024, 2),
            "used_mb": round(mem.used / 1024 / 1024, 2),
            "free_mb": round(mem.free / 1024 / 1024, 2),
            "usage_percent": round(mem.percent, 2),
            "swap": {
                "total_mb": round(swap.total / 1024 / 1024, 2),
                "used_mb": round(swap.used / 1024 / 1024, 2),
                "free_mb": round(swap.free / 1024 / 1024, 2),
                "usage_percent": round(swap.percent, 2),
            },
        }

    def get_disk_metrics(self) -> Dict[str, Any]:
        """Get disk metrics.

        Returns:
            Dictionary with disk information
        """
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()

        # Get all disk partitions
        partitions = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                partitions.append({
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "total_gb": round(usage.total / 1024 / 1024 / 1024, 2),
                    "used_gb": round(usage.used / 1024 / 1024 / 1024, 2),
                    "free_gb": round(usage.free / 1024 / 1024 / 1024, 2),
                    "usage_percent": round(usage.percent, 2),
                })
            except PermissionError:
                continue

        return {
            "root": {
                "total_gb": round(disk_usage.total / 1024 / 1024 / 1024, 2),
                "used_gb": round(disk_usage.used / 1024 / 1024 / 1024, 2),
                "free_gb": round(disk_usage.free / 1024 / 1024 / 1024, 2),
                "usage_percent": round(disk_usage.percent, 2),
            },
            "partitions": partitions,
            "io": {
                "read_mb": round(disk_io.read_bytes / 1024 / 1024, 2) if disk_io else None,
                "write_mb": round(disk_io.write_bytes / 1024 / 1024, 2) if disk_io else None,
                "read_count": disk_io.read_count if disk_io else None,
                "write_count": disk_io.write_count if disk_io else None,
            } if disk_io else None,
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all system metrics.

        Returns:
            Dictionary with all system information
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "hostname": platform.node(),
                "processor": platform.processor(),
            },
            "cpu": self.get_cpu_metrics(),
            "gpu": self.get_gpu_metrics(),
            "memory": self.get_memory_metrics(),
            "disk": self.get_disk_metrics(),
        }


# Global instance
system_monitor = SystemMonitor()
