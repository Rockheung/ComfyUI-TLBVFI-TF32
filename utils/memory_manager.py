"""
Memory management utilities for TLBVFI chunk-based processing.

Provides GPU and CPU memory cleanup functions to prevent OOM errors.
"""

import torch
import gc


def cleanup_memory(device: torch.device, force_gc: bool = True):
    """
    Clean up GPU and CPU memory.

    Args:
        device: torch.device to clean up
        force_gc: If True, force Python garbage collection
    """
    if force_gc:
        gc.collect()

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def get_memory_stats(device: torch.device) -> dict:
    """
    Get current memory usage statistics.

    Args:
        device: torch.device to query

    Returns:
        dict: Memory statistics with keys:
            - allocated_gb: Currently allocated memory in GB
            - reserved_gb: Currently reserved memory in GB
            - free_gb: Free memory in GB (CUDA only)
            - total_gb: Total memory in GB (CUDA only)
    """
    stats = {}

    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)

        try:
            free, total = torch.cuda.mem_get_info(device)
            stats['free_gb'] = free / (1024**3)
            stats['total_gb'] = total / (1024**3)
        except:
            # Fallback for older PyTorch versions
            props = torch.cuda.get_device_properties(device)
            stats['total_gb'] = props.total_memory / (1024**3)
            stats['free_gb'] = stats['total_gb'] - (reserved / (1024**3))

        stats['allocated_gb'] = allocated / (1024**3)
        stats['reserved_gb'] = reserved / (1024**3)
    else:
        stats['allocated_gb'] = 0
        stats['reserved_gb'] = 0
        stats['free_gb'] = 0
        stats['total_gb'] = 0

    return stats


def check_memory_available(device: torch.device, required_gb: float) -> bool:
    """
    Check if sufficient memory is available for processing.

    Args:
        device: torch.device to check
        required_gb: Required memory in GB

    Returns:
        bool: True if sufficient memory available
    """
    if device.type != 'cuda':
        return True  # No easy way to check CPU memory

    stats = get_memory_stats(device)
    free_gb = stats.get('free_gb', 0)

    # Use 80% of free memory as threshold for safety
    usable_gb = free_gb * 0.8

    return usable_gb >= required_gb


def estimate_frame_memory(height: int, width: int, num_frames: int = 1, dtype=torch.float32) -> float:
    """
    Estimate memory required for frames.

    Args:
        height: Frame height
        width: Frame width
        num_frames: Number of frames
        dtype: Torch data type (default: float32)

    Returns:
        float: Estimated memory in GB
    """
    bytes_per_element = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else 4
    channels = 3  # RGB

    total_bytes = height * width * channels * bytes_per_element * num_frames
    return total_bytes / (1024**3)


def print_memory_summary(device: torch.device, prefix: str = ""):
    """
    Print memory usage summary.

    Args:
        device: torch.device to query
        prefix: Optional prefix for log message
    """
    stats = get_memory_stats(device)

    if device.type == 'cuda':
        print(
            f"{prefix}GPU Memory: "
            f"{stats['allocated_gb']:.2f}GB allocated, "
            f"{stats['reserved_gb']:.2f}GB reserved, "
            f"{stats['free_gb']:.2f}GB free / {stats['total_gb']:.2f}GB total"
        )
    else:
        print(f"{prefix}Device: {device} (CPU mode)")
