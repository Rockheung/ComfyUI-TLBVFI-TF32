"""
Memory Usage Analysis for TLBVFI at 2K Resolution

This script calculates the theoretical memory usage.
"""

def analyze_memory_usage():
    # 2K resolution after padding
    H, W = 1536, 2560
    B = 1  # Batch size
    C = 24  # Number of channels at full resolution

    # VFIformer parameters
    r = 1
    n_pts = (r * 2 + 1) ** 2  # = 9

    print("=" * 80)
    print("TLBVFI Memory Analysis - 2K Resolution (2560x1536)")
    print("=" * 80)
    print()

    # Calculate memory at full resolution (rf_block1)
    print("At FULL RESOLUTION (rf_block1):")
    print(f"  Input size: {B} × {C} × {H} × {W}")
    input_size = B * C * H * W * 4  # 4 bytes for float32
    print(f"  Input memory: {input_size / 1024**2:.1f} MB")
    print()

    # After unfold
    print("After F.unfold (kernel_size=3):")
    unfold_channels = C * n_pts
    print(f"  Size: {B} × {unfold_channels} × {H} × {W}")
    unfold_size = B * unfold_channels * H * W * 4
    print(f"  Memory: {unfold_size / 1024**2:.1f} MB")
    print(f"  Memory increase: {n_pts}x (9배!)")
    print()

    # Warp operation (grid_sample needs additional memory)
    print("During warp (grid_sample):")
    print(f"  Input tensor: {unfold_size / 1024**2:.1f} MB")
    print(f"  Grid tensor: ~{B * H * W * 2 * 4 / 1024**2:.1f} MB")
    print(f"  Output tensor: {unfold_size / 1024**2:.1f} MB")
    print(f"  Temporary memory in grid_sample: ~{unfold_size / 1024**2:.1f} MB")
    warp_peak = unfold_size * 2.5  # Conservative estimate
    print(f"  Peak memory during warp: {warp_peak / 1024**2:.1f} MB")
    print()

    # Both streams (x0 and x1)
    print("Both streams (x0_unfold + x1_unfold):")
    total_unfold = warp_peak * 2
    print(f"  Peak memory: {total_unfold / 1024**2:.1f} MB")
    print()

    # Additional operations
    print("Additional operations in forward_once:")
    print("  - contents0.permute().contiguous().view()")
    print("  - contents1.permute().contiguous().view()")
    print("  - Correlation computation (einsum)")
    print("  - Flow updates")
    additional = unfold_size * 1.5
    print(f"  Additional memory: ~{additional / 1024**2:.1f} MB")
    print()

    total_peak = total_unfold + additional
    print("=" * 80)
    print(f"TOTAL PEAK MEMORY in rf_block1: {total_peak / 1024**2:.1f} MB ({total_peak / 1024**3:.2f} GB)")
    print("=" * 80)
    print()

    # Compare with different resolutions
    print("Comparison with other resolutions:")
    for res_name, (h, w) in [("1080p", (1080, 1920)), ("2K", (1536, 2560)), ("4K", (2160, 3840))]:
        mem = B * C * n_pts * h * w * 4 * 2.5 * 2  # unfold * warp_overhead * both_streams
        print(f"  {res_name} ({w}x{h}): {mem / 1024**3:.2f} GB")
    print()

    # Solution comparison
    print("=" * 80)
    print("SOLUTIONS:")
    print("=" * 80)
    print()

    print("1. Reduce flow_scale to 0.5:")
    print("   - Flow computed at 1280x768")
    print("   - Memory reduction in flow refinement: ~75%")
    print("   - Quality: Slight degradation in flow accuracy")
    print()

    print("2. Reduce r from 1 to 0:")
    print("   - n_pts: 9 → 1")
    print("   - Memory reduction: ~89%")
    print(f"   - New peak: {total_peak / 9 / 1024**3:.2f} GB")
    print("   - Quality: Moderate degradation")
    print()

    print("3. Tiled inference (512x512 tiles with 64px overlap):")
    tile_h, tile_w = 512, 512
    tile_mem = B * C * n_pts * tile_h * tile_w * 4 * 2.5 * 2
    print(f"   - Memory per tile: {tile_mem / 1024**2:.1f} MB")
    print(f"   - Memory reduction: ~{100 * (1 - tile_mem/total_peak):.1f}%")
    print("   - Quality: Potential seam artifacts at tile boundaries")
    print()

if __name__ == "__main__":
    analyze_memory_usage()
