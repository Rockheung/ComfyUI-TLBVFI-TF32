"""
Tiled Inference Utilities for High-Resolution Video Frame Interpolation

This module provides utilities to split large images into tiles with overlap,
process them independently, and blend them back together seamlessly.

This enables processing of 4K+ resolution images on GPUs with limited VRAM.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List
import math
import time

# Constants for flow_scale validation and compatibility
FLOW_SCALE_MIN = 0.05  # Minimum to avoid zero-sized tensors
FLOW_SCALE_MAX = 1.0   # Maximum (full resolution flow)
FLOW_SCALE_DEFAULT = 0.5  # Balanced speed/quality

# Network architecture constraints
# The VQGAN encoder requires input dimensions divisible by:
# min_side = 8 * 2^(num_resolutions-1) * 4 = 256 (for num_resolutions=5)
# We use a relaxed divisor for flexibility while maintaining stability
NETWORK_MIN_SIDE = 256  # Actual network requirement
TILE_SIZE_DIVISOR = 64  # Relaxed requirement (256/4) for better flexibility


def validate_and_adjust_flow_scale(flow_scale: float, tile_size: int, height: int, width: int, verbose: bool = True) -> float:
    """
    Validate flow_scale and adjust if necessary to avoid tensor size mismatches.

    This function ensures that flow_scale is compatible with the network architecture
    when used with tiling. It adjusts the scale if necessary to ensure downsampled
    tile sizes are divisible by TILE_SIZE_DIVISOR (64).

    Algorithm:
        1. Clamp flow_scale to valid range [FLOW_SCALE_MIN, FLOW_SCALE_MAX]
        2. If tile_size == 0 or flow_scale >= 1.0, return as-is (no downsampling)
        3. Calculate downsampled tile size: int(tile_size * flow_scale)
        4. Check if downsampled size is divisible by TILE_SIZE_DIVISOR
        5. If not, use mathematical optimization to find nearest compatible scale (O(1))

    Compatibility Requirements:
        The network requires downsampled images to have dimensions divisible by
        NETWORK_MIN_SIDE (256). This function uses TILE_SIZE_DIVISOR (64) as a
        relaxed requirement for flexibility while maintaining stability.

    Args:
        flow_scale: User-requested flow scale (0.0-1.0)
        tile_size: Tile size being used (0 = no tiling)
        height: Image height (currently unused, for future enhancements)
        width: Image width (currently unused, for future enhancements)
        verbose: Print warning if adjustment is made

    Returns:
        Adjusted flow_scale that ensures compatibility

    Examples:
        >>> validate_and_adjust_flow_scale(0.5, 512, 1920, 1080)
        0.5  # 512 * 0.5 = 256, divisible by 64

        >>> validate_and_adjust_flow_scale(0.3, 512, 1920, 1080, verbose=False)
        0.25  # Adjusted: 512 * 0.3 = 153.6 → 512 * 0.25 = 128, divisible by 64
    """
    # Always clamp to minimum safe value
    flow_scale = max(FLOW_SCALE_MIN, flow_scale)

    # No validation needed for full resolution flow
    if flow_scale >= FLOW_SCALE_MAX:
        return FLOW_SCALE_MAX

    # If no tiling, any scale is acceptable (handled by vqgan.py padding logic)
    if tile_size == 0:
        return flow_scale

    # Check if tile_size * flow_scale produces reasonable downsampled size
    downsampled_tile = int(tile_size * flow_scale)

    # Use the relaxed divisor for compatibility
    divisor = TILE_SIZE_DIVISOR

    # If already compatible, return as-is
    if downsampled_tile % divisor == 0:
        return flow_scale

    # Optimized: Calculate nearest compatible flow_scale mathematically (O(1) instead of O(n))
    # We want: (tile_size * candidate) % divisor == 0
    # So: candidate = (k * divisor) / tile_size, for some integer k

    # Find the k value closest to current flow_scale
    target_downsampled = flow_scale * tile_size
    k_nearest = round(target_downsampled / divisor)

    # Handle edge case: k_nearest might be 0
    if k_nearest == 0:
        k_nearest = 1

    # Calculate the mathematically optimal candidate
    candidate = (k_nearest * divisor) / tile_size

    # Clamp to valid range
    candidate = max(FLOW_SCALE_MIN, min(FLOW_SCALE_MAX, candidate))

    # Verify the candidate is actually compatible (should always be true, but double-check)
    if int(tile_size * candidate) % divisor != 0:
        # Fallback to safe default if calculation failed somehow
        if verbose:
            print(f"⚠️  Mathematical calculation failed, using {FLOW_SCALE_DEFAULT} (safe default)")
        return FLOW_SCALE_DEFAULT

    # Inform user if adjustment was made
    if verbose and abs(candidate - flow_scale) > 0.01:
        print(f"⚠️  flow_scale adjusted from {flow_scale:.3f} to {candidate:.3f} for compatibility with tile_size={tile_size}")

    return candidate


def calculate_flow_aware_tile_size(
    height: int,
    width: int,
    device: torch.device,
    flow_scale: float = 0.5,
    memory_fraction: float = 0.25,
    min_tile_size: int = 512,
    max_tile_size: int = 1536,
    step: int = 128,
    min_side: int = 256
) -> int:
    """
    Calculate optimal tile size considering both memory constraints and flow_scale compatibility.

    This function ensures that:
    1. Tile size fits in available GPU memory
    2. Downsampled tile size (tile_size * flow_scale) is compatible with network requirements

    Args:
        height: Image height in pixels
        width: Image width in pixels
        device: torch device (should be CUDA)
        flow_scale: Flow resolution scaling factor (0.05-1.0)
        memory_fraction: Fraction of available memory to use (default: 0.25 = 25%)
        min_tile_size: Minimum tile size (default: 512)
        max_tile_size: Maximum tile size (default: 1536)
        step: Tile size granularity (default: 128)
        min_side: Minimum divisible size for network (default: 256)

    Returns:
        Optimal tile size that is both memory-safe and flow_scale compatible
        Returns 0 if tiling is not needed
    """
    # First, calculate memory-based optimal tile size
    base_tile_size = calculate_optimal_tile_size(
        height, width, device, memory_fraction, min_tile_size, max_tile_size, step
    )

    # If tiling disabled, return immediately
    if base_tile_size == 0:
        return 0

    # If flow_scale >= 1.0, no downsampling, so no compatibility concern
    if flow_scale >= 1.0:
        return base_tile_size

    # Find largest tile size <= base_tile_size that is compatible with flow_scale
    # Compatible means: (tile_size * flow_scale) is divisible by TILE_SIZE_DIVISOR
    # (Matches the divisor in validate_and_adjust_flow_scale)
    divisor = TILE_SIZE_DIVISOR

    for tile_size in range(base_tile_size, min_tile_size - step, -step):
        downsampled_size = int(tile_size * flow_scale)
        if downsampled_size % divisor == 0:
            return tile_size

    # If no compatible size found in range, calculate minimum compatible size
    # Find smallest k where (k * divisor) / flow_scale >= min_tile_size
    min_downsampled = int(min_tile_size * flow_scale)
    k_min = math.ceil(min_downsampled / divisor)
    compatible_min = int((k_min * divisor) / flow_scale)

    # Round up to nearest step
    compatible_min = ((compatible_min + step - 1) // step) * step

    return compatible_min


def calculate_optimal_tile_size(
    height: int,
    width: int,
    device: torch.device,
    memory_fraction: float = 0.25,
    min_tile_size: int = 512,
    max_tile_size: int = 1536,
    step: int = 128
) -> int:
    """
    Calculate optimal tile size based on available GPU memory.

    This function estimates a safe tile size that balances speed and memory usage.
    When tile_size is set to 0, this function automatically determines the best
    tile size based on available GPU memory.

    Args:
        height: Image height in pixels
        width: Image width in pixels
        device: torch device (should be CUDA)
        memory_fraction: Fraction of available memory to use (default: 0.25 = 25%)
        min_tile_size: Minimum tile size (default: 512)
        max_tile_size: Maximum tile size (default: 1536)
        step: Tile size step/granularity (default: 128)

    Returns:
        Optimal tile size (multiple of step) between min_tile_size and max_tile_size

    Memory estimation:
        - Base model: ~3.6GB (VQGAN + UNet)
        - Per tile memory: tile_pixels * channels * dtype_size * overhead
        - Overhead factor: ~32x (very conservative for VFI model with refinement)
        - Target: Use 25% of available memory for maximum safety
    """
    if not torch.cuda.is_available() or device.type != 'cuda':
        return min_tile_size

    try:
        # Get GPU memory info
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)

        # Calculate available memory - use reserved as baseline (more accurate)
        # Add extra safety margin to avoid fragmentation issues
        used_memory = reserved_memory
        safety_margin = 3 * (1024**3)  # Reserve 3GB for safety
        available_memory = max(0, total_memory - used_memory - safety_margin)
        target_memory = available_memory * memory_fraction

        # If not enough memory available, use minimum tile size
        if target_memory < 1 * (1024**3):  # Less than 1GB available
            return min_tile_size

        # If image is small enough to fit in memory without tiling
        if height <= min_tile_size and width <= min_tile_size:
            return 0

        # Memory estimation formula (very conservative):
        # Per-pixel memory = channels (3) * dtype_size (4 for FP32) * overhead (32x)
        # Very high overhead multiplier for VFI models with:
        # - Encoder features (multi-scale)
        # - Decoder features (multi-scale)
        # - Optical flow computation and refinement (VFIformer)
        # - Attention maps
        # - Intermediate tensors
        # - Gradient buffers
        channels = 3
        dtype_size = 4
        overhead_multiplier = 32  # Very conservative for VFI
        bytes_per_pixel = channels * dtype_size * overhead_multiplier

        # Calculate maximum safe tile size
        max_tile_pixels = target_memory / bytes_per_pixel
        max_safe_tile_size = int(math.sqrt(max_tile_pixels))

        # Round down to nearest step
        optimal_tile_size = (max_safe_tile_size // step) * step

        # Clamp to valid range
        optimal_tile_size = max(min_tile_size, min(optimal_tile_size, max_tile_size))

        # Additional safety check: if optimal size covers the whole image, disable tiling
        if optimal_tile_size >= height and optimal_tile_size >= width:
            return 0

        return optimal_tile_size

    except Exception as e:
        return min_tile_size


def calculate_tiles(height: int, width: int, tile_size: int = 512, overlap: int = 64) -> List[Tuple[int, int, int, int]]:
    """
    Calculate tile coordinates for a given image size.

    Args:
        height: Image height
        width: Image width
        tile_size: Size of each tile (default: 512)
        overlap: Overlap between adjacent tiles (default: 64)

    Returns:
        List of (y_start, y_end, x_start, x_end) tuples
    """
    tiles = []
    stride = tile_size - overlap

    # Calculate number of tiles needed
    n_tiles_h = math.ceil((height - overlap) / stride)
    n_tiles_w = math.ceil((width - overlap) / stride)

    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            y_start = i * stride
            x_start = j * stride

            # Last tile might be smaller, so adjust
            y_end = min(y_start + tile_size, height)
            x_end = min(x_start + tile_size, width)

            # If tile is too small, extend it backwards
            if y_end - y_start < tile_size and y_start > 0:
                y_start = max(0, y_end - tile_size)
            if x_end - x_start < tile_size and x_start > 0:
                x_start = max(0, x_end - tile_size)

            tiles.append((y_start, y_end, x_start, x_end))

    return tiles


def create_blend_mask(tile_h: int, tile_w: int, overlap: int, device: torch.device) -> torch.Tensor:
    """
    Create a blending mask for seamless tile merging.

    The mask uses distance-based weighting from tile edges.
    Each pixel's weight = min(distance_from_top, distance_from_bottom,
                               distance_from_left, distance_from_right, overlap) / overlap

    Args:
        tile_h: Tile height
        tile_w: Tile width
        overlap: Overlap size
        device: torch device

    Returns:
        Blending mask of shape (1, 1, tile_h, tile_w)
    """
    if overlap == 0:
        return torch.ones((1, 1, tile_h, tile_w), device=device)

    # Create coordinate grids
    y_coords = torch.arange(tile_h, device=device).float()
    x_coords = torch.arange(tile_w, device=device).float()

    # Distance from each edge
    dist_from_top = y_coords.view(-1, 1)
    dist_from_bottom = (tile_h - 1 - y_coords).view(-1, 1)
    dist_from_left = x_coords.view(1, -1)
    dist_from_right = (tile_w - 1 - x_coords).view(1, -1)

    # Minimum distance from any edge (this determines the blend weight)
    mask = torch.minimum(
        torch.minimum(dist_from_top, dist_from_bottom),
        torch.minimum(dist_from_left, dist_from_right)
    )

    # Clamp to overlap distance and normalize
    mask = torch.clamp(mask, max=overlap) / overlap

    return mask.unsqueeze(0).unsqueeze(0)


def extract_tile(image: torch.Tensor, tile_coords: Tuple[int, int, int, int]) -> torch.Tensor:
    """
    Extract a tile from an image.

    Args:
        image: Input image tensor (B, C, H, W)
        tile_coords: (y_start, y_end, x_start, x_end)

    Returns:
        Tile tensor (B, C, tile_h, tile_w)
    """
    y_start, y_end, x_start, x_end = tile_coords
    return image[:, :, y_start:y_end, x_start:x_end].contiguous()


def merge_tiles(tiles: List[torch.Tensor],
                tile_coords_list: List[Tuple[int, int, int, int]],
                output_shape: Tuple[int, int, int, int],
                overlap: int,
                device: torch.device) -> torch.Tensor:
    """
    Merge tiles back into a full image with blending.

    Args:
        tiles: List of tile tensors
        tile_coords_list: List of tile coordinates
        output_shape: (B, C, H, W) shape of output image
        overlap: Overlap size used
        device: torch device

    Returns:
        Merged image tensor (B, C, H, W)
    """
    B, C, H, W = output_shape
    output = torch.zeros(output_shape, device=device, dtype=tiles[0].dtype)
    weight_sum = torch.zeros((1, 1, H, W), device=device, dtype=tiles[0].dtype)

    for tile, coords in zip(tiles, tile_coords_list):
        y_start, y_end, x_start, x_end = coords
        tile_h, tile_w = y_end - y_start, x_end - x_start

        # Create blend mask for this tile
        mask = create_blend_mask(tile_h, tile_w, overlap, device)

        # Add weighted tile to output
        output[:, :, y_start:y_end, x_start:x_end] += tile * mask
        weight_sum[:, :, y_start:y_end, x_start:x_end] += mask

    # Normalize by weight sum to get final blended result
    output = output / weight_sum.clamp(min=1e-8)

    return output


def process_with_tiling(model,
                       frame_a: torch.Tensor,
                       frame_b: torch.Tensor,
                       tile_size: int = 512,
                       overlap: int = 64,
                       scale: float = 1.0,
                       debug: bool = False) -> torch.Tensor:
    """
    Process two frames through a model using tiled inference.

    This function splits the input frames into tiles, processes each tile
    independently through the model, and blends the results back together.

    Args:
        model: The interpolation model with a .sample() method
        frame_a: First frame tensor (B, C, H, W)
        frame_b: Second frame tensor (B, C, H, W)
        tile_size: Size of each tile (default: 512)
        overlap: Overlap between tiles (default: 64)
        scale: Flow scale parameter for model

    Returns:
        Interpolated frame tensor (B, C, H, W)
    """
    B, C, H, W = frame_a.shape
    device = frame_a.device
    dtype = frame_a.dtype

    # Validate tile_size
    if tile_size < overlap * 2:
        raise ValueError(
            f"tile_size ({tile_size}) must be at least 2x overlap ({overlap * 2}). "
            f"Recommended: tile_size >= 256 for proper tiling. "
            f"Use tile_size=0 to disable tiling."
        )

    # If image is smaller than tile size, process directly
    if H <= tile_size and W <= tile_size:
        return model.sample(frame_a, frame_b, scale=scale)

    # Calculate tile coordinates
    tile_coords_list = calculate_tiles(H, W, tile_size, overlap)

    if debug:
        print(f"  Tiled processing: {len(tile_coords_list)} tiles ({tile_size}x{tile_size}, overlap={overlap})")

    # Process each tile with progress tracking
    output_tiles = []
    total_tiles = len(tile_coords_list)
    start_time = time.time()
    tile_times = []

    for i, coords in enumerate(tile_coords_list):
        try:
            tile_start_time = time.time()

            # Extract tiles from both frames
            tile_a = extract_tile(frame_a, coords)
            tile_b = extract_tile(frame_b, coords)

            # Debug info for first tile
            if debug and i == 0:
                print(f"    First tile shape: {tile_a.shape}")

            # Process tile through model
            with torch.no_grad():
                tile_out = model.sample(tile_a, tile_b, scale=scale)

            output_tiles.append(tile_out)

            # Track tile processing time
            tile_elapsed = time.time() - tile_start_time
            tile_times.append(tile_elapsed)

            # Clear cache periodically
            if (i + 1) % 4 == 0:
                torch.cuda.empty_cache()

            # Calculate progress and ETA (only in debug mode)
            if debug:
                progress_pct = ((i + 1) / total_tiles) * 100

                # Calculate ETA after processing at least 2 tiles
                if len(tile_times) >= 2:
                    avg_time_per_tile = sum(tile_times) / len(tile_times)
                    remaining_tiles = total_tiles - (i + 1)
                    eta_seconds = avg_time_per_tile * remaining_tiles

                    # Format ETA
                    if eta_seconds < 60:
                        eta_str = f"{eta_seconds:.0f}s"
                    elif eta_seconds < 3600:
                        eta_str = f"{eta_seconds/60:.1f}m"
                    else:
                        eta_str = f"{eta_seconds/3600:.1f}h"

                    # Progress bar
                    bar_length = 30
                    filled_length = int(bar_length * (i + 1) / total_tiles)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)

                    print(f"\r    [{bar}] {i+1}/{total_tiles} ({progress_pct:.1f}%) | ETA: {eta_str}", end='', flush=True)
                else:
                    # Simple progress for first tiles
                    bar_length = 30
                    filled_length = int(bar_length * (i + 1) / total_tiles)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    print(f"\r    [{bar}] {i+1}/{total_tiles} ({progress_pct:.1f}%)", end='', flush=True)

        except Exception as e:
            print(f"\n    ERROR processing tile {i+1}/{len(tile_coords_list)}: {e}")
            print(f"    Tile coords: {coords}")
            print(f"    Tile shape: {tile_a.shape if 'tile_a' in locals() else 'N/A'}")
            raise

    # Print newline after progress bar (only in debug mode)
    if debug:
        print()

        # Print total time
        total_time = time.time() - start_time
        if total_time < 60:
            time_str = f"{total_time:.1f}s"
        elif total_time < 3600:
            time_str = f"{total_time/60:.1f}m"
        else:
            time_str = f"{total_time/3600:.1f}h"
        print(f"    Completed in {time_str}")

    # Verify we have tiles before merging
    if not output_tiles:
        raise RuntimeError(f"No tiles were successfully processed! Total tiles: {len(tile_coords_list)}")

    # Merge tiles back together with blending
    output = merge_tiles(output_tiles, tile_coords_list, (B, C, H, W), overlap, device)

    # Clean up
    del output_tiles
    torch.cuda.empty_cache()

    return output


def test_tiling():
    """Test the tiling functions with a simple example."""
    print("Testing tiling utilities...")

    # Test calculate_tiles
    tiles = calculate_tiles(2160, 3840, tile_size=512, overlap=64)
    print(f"\n4K resolution (3840x2160) with 512x512 tiles, 64px overlap:")
    print(f"  Number of tiles: {len(tiles)}")
    print(f"  First tile: {tiles[0]}")
    print(f"  Last tile: {tiles[-1]}")

    # Test blend mask
    device = torch.device('cpu')
    mask = create_blend_mask(512, 512, 64, device)
    print(f"\nBlend mask shape: {mask.shape}")
    print(f"  Top-left corner (overlap area): {mask[0, 0, 0, 0]:.3f}")
    print(f"  Center (no overlap): {mask[0, 0, 256, 256]:.3f}")

    # Test tile extraction and merging
    test_img = torch.randn(1, 3, 1080, 1920, device=device)
    tiles_coords = calculate_tiles(1080, 1920, tile_size=512, overlap=64)

    # Extract tiles
    tiles = [extract_tile(test_img, coords) for coords in tiles_coords]

    # Merge back
    merged = merge_tiles(tiles, tiles_coords, test_img.shape, overlap=64, device=device)

    # Check if merging is close to original
    diff = (test_img - merged).abs().max()
    print(f"\n1080p test:")
    print(f"  Number of tiles: {len(tiles)}")
    print(f"  Max difference after merge: {diff:.6f}")
    print(f"  Test {'PASSED' if diff < 1e-5 else 'FAILED'}")

    print("\nTiling utilities test complete!")


if __name__ == "__main__":
    test_tiling()
