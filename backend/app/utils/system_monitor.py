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
        """Check if GPU monitoring is available.

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
            return False
        except Exception:
            return False

    def get_cpu_metrics(self) -> Dict[str, Any]:
        """Get CPU metrics.

        Returns:
            Dictionary with CPU information
        """
        # Use single measurement with interval=0.1 for faster response
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_freq = psutil.cpu_freq()

        # Calculate average usage from per-core data (more accurate)
        avg_usage = sum(cpu_percent) / len(cpu_percent) if cpu_percent else 0.0

        return {
            "usage_percent": round(avg_usage, 2),
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

        Follows official pynvml pattern: nvmlInit() -> operations -> nvmlShutdown()

        Returns:
            List of GPU information dictionaries
        """
        if not self.gpu_available:
            return []

        try:
            import pynvml
            pynvml.nvmlInit()

            try:
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

                # Shutdown NVML (official pattern requires cleanup)
                pynvml.nvmlShutdown()
                return gpu_info

            except Exception as inner_e:
                # Ensure shutdown even on error
                try:
                    pynvml.nvmlShutdown()
                except:
                    pass
                raise inner_e

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


    def get_optimal_batch_size(
        self,
        image_size: tuple = (640, 640),
        model_type: str = "yolo",
        safety_margin: float = 0.7,  # INCREASED: More conservative to prevent OOM
        default_batch_size: int = 128  # REDUCED: Conservative default
    ) -> int:
        """
        Calculate optimal batch size based on available GPU memory.

        IMPORTANT: Conservative estimation to prevent OOM errors.
        Real memory usage is often higher due to:
        - Model weights already loaded on GPU
        - PyTorch CUDA cache
        - Gradient accumulation
        - Intermediate activations
        - EOT sampling (for patch attacks)

        Args:
            image_size: Input image size (height, width)
            model_type: Model type ("yolo", "patch", "noise")
            safety_margin: Safety margin (0.0-0.9) to prevent OOM errors (default: 0.7)
            default_batch_size: Default batch size if GPU not available or calculation fails

        Returns:
            Optimal batch size
        """
        if not self.gpu_available:
            return default_batch_size

        try:
            import pynvml
            pynvml.nvmlInit()

            # Get first GPU (index 0) - most attacks run on single GPU
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # CRITICAL: Use TOTAL memory, not free memory
            # Model is already loaded, so we work with remaining capacity
            total_mb = mem_info.total / 1024 / 1024
            used_mb = mem_info.used / 1024 / 1024
            free_mb = mem_info.free / 1024 / 1024

            # Calculate available memory for batch processing
            # Use smaller of: (free memory) or (total * (1-safety_margin) - used)
            # This accounts for memory already in use
            max_usable_mb = min(
                free_mb * 0.8,  # Use only 80% of free memory
                total_mb * (1 - safety_margin) - used_mb  # Reserve safety margin from total
            )

            pynvml.nvmlShutdown()

            # Estimate memory per image based on image size and model type
            h, w = image_size
            pixels = h * w
            base_mb_per_image = (pixels * 3 * 4) / (1024 * 1024)  # 4 bytes per float32

            # Memory estimation (MB per image) - CONSERVATIVE values
            # Based on empirical testing - these are MINIMUM estimates
            if model_type == "yolo":
                # YOLOv8/v11 detection + gradients
                mb_per_image = base_mb_per_image * 8  # INCREASED: 4 → 8
            elif model_type == "patch":
                # Adversarial patch generation (VERY memory intensive)
                # Transformations + EOT samples (5x) + gradients + optimizer states
                mb_per_image = base_mb_per_image * 15  # INCREASED: 6 → 15
            elif model_type == "noise":
                # Noise attack (moderately memory intensive)
                mb_per_image = base_mb_per_image * 6  # INCREASED: 3 → 6
            else:
                mb_per_image = base_mb_per_image * 8

            # Calculate optimal batch size
            if mb_per_image > 0 and max_usable_mb > 0:
                optimal_batch_size = int(max_usable_mb / mb_per_image)
                # Clamp between reasonable bounds - REDUCED max from 256 to 64
                optimal_batch_size = max(1, min(optimal_batch_size, 64))
            else:
                optimal_batch_size = default_batch_size

            print(f"[GPU Memory] Total: {total_mb:.1f} MB, Used: {used_mb:.1f} MB, "
                  f"Free: {free_mb:.1f} MB")
            print(f"[GPU Memory] Max usable: {max_usable_mb:.1f} MB, "
                  f"Est. {mb_per_image:.1f} MB/image")
            print(f"[GPU Memory] Optimal batch size: {optimal_batch_size}")

            return optimal_batch_size

        except Exception as e:
            print(f"[GPU Memory] Failed to calculate optimal batch size: {e}")
            import traceback
            traceback.print_exc()
            return default_batch_size


# Global instance
system_monitor = SystemMonitor()
