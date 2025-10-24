"""
Test script for automatic tile size calculation.

This script tests the calculate_optimal_tile_size function with various
resolution scenarios to verify it works correctly.
"""

import torch
import math
from typing import Tuple, List


def calculate_optimal_tile_size(
    height: int,
    width: int,
    device: torch.device,
    memory_fraction: float = 0.8,
    min_tile_size: int = 512,
    max_tile_size: int = 2048,
    step: int = 128
) -> int:
    """
    Calculate optimal tile size based on available GPU memory.
    (Copied from utils/tiling.py for standalone testing)
    """
    if not torch.cuda.is_available() or device.type != 'cuda':
        print(f"  Auto tile size: CPU detected, using minimum ({min_tile_size})")
        return min_tile_size

    try:
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)

        used_memory = max(allocated_memory, reserved_memory)
        available_memory = total_memory - used_memory
        target_memory = available_memory * memory_fraction

        total_gb = total_memory / (1024**3)
        available_gb = available_memory / (1024**3)
        target_gb = target_memory / (1024**3)

        print(f"\n{'='*80}")
        print(f"Auto Tile Size Calculation:")
        print(f"  GPU Memory:")
        print(f"    Total: {total_gb:.2f}GB")
        print(f"    Available: {available_gb:.2f}GB")
        print(f"    Target (80%): {target_gb:.2f}GB")
        print(f"  Image Resolution: {height}x{width}")

        image_pixels = height * width
        if height <= min_tile_size and width <= min_tile_size:
            print(f"  Decision: Image small enough ({height}x{width}), disable tiling")
            print(f"{'='*80}\n")
            return 0

        channels = 3
        dtype_size = 4
        overhead_multiplier = 8
        bytes_per_pixel = channels * dtype_size * overhead_multiplier

        max_tile_pixels = target_memory / bytes_per_pixel
        max_safe_tile_size = int(math.sqrt(max_tile_pixels))

        optimal_tile_size = (max_safe_tile_size // step) * step
        optimal_tile_size = max(min_tile_size, min(optimal_tile_size, max_tile_size))

        if optimal_tile_size >= height and optimal_tile_size >= width:
            print(f"  Calculated: {optimal_tile_size}x{optimal_tile_size}")
            print(f"  Decision: Tile size covers full image, disable tiling")
            print(f"{'='*80}\n")
            return 0

        print(f"  Estimated memory per tile:")
        print(f"    Tile size: {optimal_tile_size}x{optimal_tile_size}")
        tile_memory_gb = (optimal_tile_size ** 2 * bytes_per_pixel) / (1024**3)
        print(f"    Memory: {tile_memory_gb:.2f}GB")
        print(f"  Decision: Use {optimal_tile_size}x{optimal_tile_size} tiles")
        print(f"{'='*80}\n")

        return optimal_tile_size

    except Exception as e:
        print(f"  Warning: Failed to calculate optimal tile size: {e}")
        print(f"  Fallback: Using minimum tile size ({min_tile_size})")
        return min_tile_size


def test_auto_tile_size():
    """Test automatic tile size calculation with different resolutions."""

    print("=" * 80)
    print("Testing Automatic Tile Size Calculation")
    print("=" * 80)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("\nWARNING: CUDA not available. Testing with CPU fallback.")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        print(f"\nUsing device: {device}")

        # Print GPU info
        props = torch.cuda.get_device_properties(device)
        total_memory_gb = props.total_memory / (1024**3)
        print(f"GPU: {props.name}")
        print(f"Total Memory: {total_memory_gb:.2f}GB")

    # Test scenarios with different resolutions
    test_cases = [
        ("480p (SD)", 854, 480),
        ("720p (HD)", 1280, 720),
        ("1080p (Full HD)", 1920, 1080),
        ("1440p (2K)", 2560, 1440),
        ("2160p (4K)", 3840, 2160),
        ("4320p (8K)", 7680, 4320),
    ]

    print("\n" + "=" * 80)
    print("Test Results:")
    print("=" * 80)

    for name, width, height in test_cases:
        print(f"\n{'-' * 80}")
        print(f"Test Case: {name} ({width}x{height})")
        print(f"{'-' * 80}")

        tile_size = calculate_optimal_tile_size(
            height=height,
            width=width,
            device=device,
            memory_fraction=0.8,
            min_tile_size=512,
            max_tile_size=2048,
            step=128
        )

        if tile_size == 0:
            print(f"Result: Tiling DISABLED (process full image)")
        else:
            print(f"Result: Use {tile_size}x{tile_size} tiles")

            # Estimate how many tiles would be needed
            overlap = 64
            stride = tile_size - overlap
            n_tiles_h = math.ceil((height - overlap) / stride)
            n_tiles_w = math.ceil((width - overlap) / stride)
            num_tiles = n_tiles_h * n_tiles_w
            print(f"Estimated number of tiles: {num_tiles} ({n_tiles_h}x{n_tiles_w})")

    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)

    # Test with simulated memory pressure
    if torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("Testing with Simulated Memory Pressure")
        print("=" * 80)

        # Allocate some memory to simulate model being loaded
        print("\nAllocating 4GB to simulate loaded model...")
        dummy_tensor = torch.randn(1024, 1024, 1024, dtype=torch.float32, device=device)  # ~4GB

        print(f"\nMemory allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f}GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(device) / (1024**3):.2f}GB")

        print("\nRecalculating for 4K with memory pressure:")
        tile_size = calculate_optimal_tile_size(
            height=2160,
            width=3840,
            device=device,
            memory_fraction=0.8,
            min_tile_size=512,
            max_tile_size=2048,
            step=128
        )

        if tile_size == 0:
            print(f"Result: Tiling DISABLED")
        else:
            print(f"Result: Use {tile_size}x{tile_size} tiles")

        # Clean up
        del dummy_tensor
        torch.cuda.empty_cache()

        print("\n" + "=" * 80)


if __name__ == "__main__":
    test_auto_tile_size()
