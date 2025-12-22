"""
System statistics monitoring service.
Provides real-time CPU, GPU, memory, and disk usage stats.
"""
import psutil
import os
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
        # Keep track of the original procfs path (psutil default)
        self.procfs_default = getattr(psutil, "PROCFS_PATH", "/proc")
        self.procfs_host = None
        self.procfs_has_net = False

        # Allow overriding procfs path (e.g., host /proc mounted at /host/proc inside container)
        procfs_path = os.getenv("PROCFS_PATH")
        if procfs_path and os.path.exists(procfs_path):
            try:
                psutil.PROCFS_PATH = procfs_path
                self.procfs_host = procfs_path
                self.procfs_has_net = os.path.exists(os.path.join(procfs_path, "net", "dev"))
                if not self.procfs_has_net:
                    logger.warning(f"PROCFS_PATH set to {procfs_path} but net/dev not found; network stats will fallback")
                logger.info(f"Using custom PROCFS_PATH: {procfs_path}")
            except Exception as e:
                logger.warning(f"Failed to set PROCFS_PATH to {procfs_path}: {e}")

        # Allow overriding cgroup path (e.g., host /sys/fs/cgroup mounted at /host/sys/fs/cgroup)
        cgroup_env = os.getenv("CGROUPFS_PATH")
        default_cgroup = "/sys/fs/cgroup"
        if cgroup_env and os.path.exists(cgroup_env):
            self.cgroup_base = cgroup_env.rstrip("/")
            logger.info(f"Using custom CGROUPFS_PATH: {self.cgroup_base}")
        else:
            self.cgroup_base = default_cgroup

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
        """Check if GPU monitoring is available (using pynvml).

        Follows official pynvml pattern: nvmlInit() -> check -> nvmlShutdown()
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                has_gpu = device_count > 0
            finally:
                # Always shutdown after check (official pattern)
                pynvml.nvmlShutdown()
            return has_gpu
        except ImportError:
            logger.warning("pynvml (nvidia-ml-py) not installed")
            return False
        except pynvml.NVMLError_LibraryNotFound:
            logger.warning("NVIDIA driver not found")
            return False
        except Exception as e:
            logger.warning(f"pynvml check failed: {e}")
            return False

    def _cgroup_path(self, *parts: str) -> str:
        """Build a path inside the (possibly overridden) cgroup filesystem."""
        return os.path.join(self.cgroup_base, *parts)

    def _read_cgroup_value(self, path: str) -> Optional[str]:
        """Read a value from cgroup files safely."""
        try:
            with open(path, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.debug(f"Failed to read {path}: {e}")
            return None

    def _get_cgroup_cpu_limit(self) -> Optional[float]:
        """
        Determine CPU quota applied by cgroups.

        Returns:
            Number of CPUs available to the container/process (can be fractional) or None if unlimited.
        """
        # cgroup v2: cpu.max => "<quota> <period>", quota can be "max"
        cpu_max = self._read_cgroup_value(self._cgroup_path("cpu.max"))
        if cpu_max:
            try:
                quota_str, period_str = cpu_max.split()
                if quota_str != "max":
                    quota = float(quota_str)
                    period = float(period_str)
                    if quota > 0 and period > 0:
                        return quota / period
            except Exception as e:
                logger.debug(f"Failed to parse cpu.max: {e}")

        # cgroup v1: cpu.cfs_quota_us / cpu.cfs_period_us
        quota_str = self._read_cgroup_value(self._cgroup_path("cpu", "cpu.cfs_quota_us"))
        period_str = self._read_cgroup_value(self._cgroup_path("cpu", "cpu.cfs_period_us"))
        try:
            if quota_str is not None and period_str is not None:
                quota = float(quota_str)
                period = float(period_str)
                if quota > 0 and period > 0:
                    return quota / period
        except Exception as e:
            logger.debug(f"Failed to parse cgroup v1 cpu quota: {e}")

        return None

    def _get_affinity_cpu_indices(self) -> Optional[list[int]]:
        """Return CPU indices available to this process via affinity/cpuset."""
        try:
            process = psutil.Process()
            if hasattr(process, "cpu_affinity"):
                return process.cpu_affinity()
        except Exception as e:
            logger.debug(f"cpu_affinity read failed: {e}")
        return None

    def get_cpu_stats(self) -> Dict:
        """Get CPU usage statistics."""
        # Sample per-core usage once (single measurement to avoid delay)
        try:
            # Use interval=0.1 for faster response while still getting accurate reading
            cpu_percent_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            cpu_percent_total = None
        except Exception as e:
            logger.warning(f"cpu_percent(percpu=True) failed, falling back to aggregate: {e}")
            cpu_percent_per_core = []
            cpu_percent_total = psutil.cpu_percent(interval=0.1)

        # Try to get CPU frequency (may fail on some systems like macOS)
        try:
            cpu_freq = psutil.cpu_freq()
        except (OSError, FileNotFoundError):
            cpu_freq = None

        logical_cpus = psutil.cpu_count(logical=True) or len(cpu_percent_per_core) or 1
        affinity_cpus = self._get_affinity_cpu_indices()
        cgroup_cpu_limit = self._get_cgroup_cpu_limit()

        # If affinity is set, trim per-core usage to the CPUs we can actually run on
        if affinity_cpus and cpu_percent_per_core:
            usage_per_core = [cpu_percent_per_core[i] for i in affinity_cpus if i < len(cpu_percent_per_core)]
        else:
            usage_per_core = cpu_percent_per_core if cpu_percent_per_core else ([cpu_percent_total] if cpu_percent_total is not None else [])

        # Compute effective CPU capacity (respect cgroup quota and affinity if present)
        effective_cpu_capacity = float(logical_cpus)
        if affinity_cpus:
            effective_cpu_capacity = min(effective_cpu_capacity, float(len(affinity_cpus)))
        if cgroup_cpu_limit:
            effective_cpu_capacity = min(effective_cpu_capacity, float(cgroup_cpu_limit))

        # Avoid division by zero
        effective_cpu_capacity = max(effective_cpu_capacity, 1e-3)

        # Calculate normalized usage
        # In container environments, we want to show usage relative to available CPUs
        # If we have 4 cores available and are using 2 fully, show 50% not 200%
        if usage_per_core:
            # Average usage across available cores (more intuitive than sum)
            avg_usage = sum(usage_per_core) / len(usage_per_core) if usage_per_core else 0.0

            # If we're limited by cgroup/affinity, scale accordingly
            if len(usage_per_core) < effective_cpu_capacity:
                # Some cores are restricted, use average as-is
                normalized_usage = min(avg_usage, 100.0)
            else:
                # All cores available, average gives us the right percentage
                normalized_usage = min(avg_usage, 100.0)
        else:
            # Fallback to aggregate if per-core is unavailable
            normalized_usage = cpu_percent_total if cpu_percent_total is not None else 0.0

        return {
            "usage_percent": round(normalized_usage, 2),
            "usage_per_core": [round(p, 2) for p in usage_per_core] if usage_per_core else None,
            "core_count": psutil.cpu_count(logical=False),
            "thread_count": psutil.cpu_count(logical=True),
            "frequency_mhz": round(cpu_freq.current, 2) if cpu_freq else None,
            "frequency_max_mhz": round(cpu_freq.max, 2) if cpu_freq else None,
            "cpu_capacity": round(effective_cpu_capacity, 2),
        }

    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics.

        Uses psutil.virtual_memory() which provides accurate memory statistics.
        Note: mem.used includes buffers/cache on Linux but psutil's mem.percent
        is calculated as: (total - available) / total * 100
        This gives a more accurate representation of actual memory pressure.
        """
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Calculate actual used memory (total - available is more accurate than mem.used)
        # This accounts for buffers/cache which can be freed if needed
        actual_used_gb = (mem.total - mem.available) / (1024**3)

        # Ensure percentages don't exceed 100% due to rounding
        memory_percent = min(mem.percent, 100.0)
        swap_percent = min(swap.percent, 100.0)

        return {
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "used_gb": round(actual_used_gb, 2),  # Use calculated value instead of mem.used
            "percent": round(memory_percent, 2),
            "swap_total_gb": round(swap.total / (1024**3), 2),
            "swap_used_gb": round(swap.used / (1024**3), 2),
            "swap_percent": round(swap_percent, 2),
        }

    def get_disk_stats(self) -> Dict:
        """Get disk usage statistics."""
        disk = psutil.disk_usage('/')
        io = psutil.disk_io_counters()

        # Ensure percentage doesn't exceed 100% due to reserved space or rounding
        disk_percent = min(disk.percent, 100.0)

        return {
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent": round(disk_percent, 2),
            "read_mb": round(io.read_bytes / (1024**2), 2) if io else None,
            "write_mb": round(io.write_bytes / (1024**2), 2) if io else None,
        }

    def _get_gpu_stats_pynvml(self) -> Optional[Dict]:
        """Get GPU stats using pynvml (NVIDIA Management Library).

        Follows official pynvml pattern: nvmlInit() -> operations -> nvmlShutdown()
        https://pythonhosted.org/nvidia-ml-py/
        """
        try:
            import pynvml

            # Initialize NVML (official pattern requires init before operations)
            pynvml.nvmlInit()

            try:
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

                        # Safe percentage calculation with validation
                        if memory_total_mb > 0:
                            memory_percent = min((memory_used_mb / memory_total_mb * 100), 100.0)
                        else:
                            memory_percent = 0.0
                    except pynvml.NVMLError as e:
                        # Method 2: Try BAR1 memory (for virtual GPUs)
                        try:
                            bar1_mem = pynvml.nvmlDeviceGetBAR1MemoryInfo(handle)
                            memory_total_mb = bar1_mem.bar1Total / (1024 ** 2)
                            memory_used_mb = bar1_mem.bar1Used / (1024 ** 2)
                            memory_free_mb = bar1_mem.bar1Free / (1024 ** 2)

                            # Safe percentage calculation with validation
                            if memory_total_mb > 0:
                                memory_percent = min((memory_used_mb / memory_total_mb * 100), 100.0)
                            else:
                                memory_percent = 0.0
                        except pynvml.NVMLError as e2:
                            logger.debug(f"GPU {i}: Failed to get memory info (standard and BAR1): {e}, {e2}")
                            pass

                    # Get utilization (GPU load)
                    load_percent = None
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        load_percent = float(util.gpu)
                    except pynvml.NVMLError as e:
                        pass

                    # Get temperature
                    temp = None
                    try:
                        temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                    except pynvml.NVMLError as e:
                        pass

                    # Get power usage
                    power_watts = None
                    try:
                        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                        power_watts = power_mw / 1000.0  # Convert milliwatts to watts
                    except pynvml.NVMLError as e:
                        pass

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

                result = {
                    "available": True,
                    "count": len(gpu_stats),
                    "gpus": gpu_stats,
                }

                # Shutdown NVML (official pattern requires cleanup)
                pynvml.nvmlShutdown()
                return result

            except Exception as inner_e:
                # Ensure shutdown even on error
                try:
                    pynvml.nvmlShutdown()
                except:
                    pass
                raise inner_e

        except ImportError:
            logger.error("pynvml is not installed")
            return None
        except pynvml.NVMLError_LibraryNotFound:
            logger.warning("NVIDIA driver not found")
            return None
        except Exception as e:
            logger.error(f"pynvml failed: {e}")
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

                if mem_used_elem is not None:
                    memory_used_mb = self._parse_float_safe(mem_used_elem.text, 0.0)

                if mem_free_elem is not None:
                    memory_free_mb = self._parse_float_safe(mem_free_elem.text, 0.0)

                # If fb_memory_usage failed, try bar1_memory_usage
                if memory_total_mb == 0.0:
                    mem_total_elem = gpu.find('.//bar1_memory_usage/total')
                    mem_used_elem = gpu.find('.//bar1_memory_usage/used')
                    mem_free_elem = gpu.find('.//bar1_memory_usage/free')

                    if mem_total_elem is not None:
                        memory_total_mb = self._parse_float_safe(mem_total_elem.text, 0.0)

                    if mem_used_elem is not None:
                        memory_used_mb = self._parse_float_safe(mem_used_elem.text, 0.0)

                    if mem_free_elem is not None:
                        memory_free_mb = self._parse_float_safe(mem_free_elem.text, 0.0)

                memory_percent = (memory_used_mb / memory_total_mb * 100) if memory_total_mb > 0 else 0

                # Get temperature
                temp_elem = gpu.find('.//temperature/gpu_temp')
                temperature_c = self._parse_float_safe(temp_elem.text if temp_elem is not None else None, 0.0)

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
            return None

    def get_gpu_stats(self) -> Optional[Dict]:
        """Get GPU usage statistics using pynvml (NVIDIA Management Library)."""
        if not self.gpu_available:
            return None

        # Use pynvml (NVIDIA Management Library)
        result = self._get_gpu_stats_pynvml()
        if result is not None:
            return result

        # If failed
        logger.error("pynvml failed to get GPU stats")
        return {
            "available": False,
            "error": "Unable to get GPU stats from pynvml"
        }

    def get_network_stats(self) -> Dict:
        """Get network I/O statistics."""
        # psutil reads /proc/net/dev; if host procfs lacks it, fall back to default procfs
        original_procfs = getattr(psutil, "PROCFS_PATH", "/proc")
        try:
            if self.procfs_host and self.procfs_has_net:
                psutil.PROCFS_PATH = self.procfs_host
            else:
                psutil.PROCFS_PATH = self.procfs_default

            net = psutil.net_io_counters()
        except FileNotFoundError as e:
            logger.warning(f"Network stats fallback to default procfs due to missing net/dev: {e}")
            psutil.PROCFS_PATH = self.procfs_default
            net = psutil.net_io_counters()
        finally:
            # Restore whichever procfs was active before this call
            psutil.PROCFS_PATH = original_procfs

        return {
            "bytes_sent_mb": round(net.bytes_sent / (1024**2), 2),
            "bytes_recv_mb": round(net.bytes_recv / (1024**2), 2),
            "packets_sent": net.packets_sent,
            "packets_recv": net.packets_recv,
        }

    def get_process_stats(self) -> Dict:
        """Get current process statistics."""
        # When PROCFS_PATH points to host /proc, our container PID may not exist there.
        # Temporarily switch to the original procfs to read current process stats safely.
        original_procfs = getattr(psutil, "PROCFS_PATH", "/proc")
        try:
            psutil.PROCFS_PATH = self.procfs_default
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
        finally:
            psutil.PROCFS_PATH = original_procfs

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
