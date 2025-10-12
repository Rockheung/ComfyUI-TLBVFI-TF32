"""
FramePairSlicer Node for TLBVFI Chunk-Based Workflow

Extracts consecutive frame pairs from video for sequential interpolation processing.
This node is the first step in the chunk-based workflow that enables processing
long 4K videos without memory exhaustion.
"""

import torch


class FramePairSlicer:
    """
    Extract consecutive frame pairs from IMAGE batch for interpolation.

    This node enables chunk-based processing by slicing video into frame pairs
    that can be interpolated independently.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # (N, H, W, C) from VHS LoadVideo
                "pair_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("frame_pair", "pair_index", "total_pairs", "is_last_pair")
    FUNCTION = "slice_pair"
    CATEGORY = "frame_interpolation/TLBVFI-TF32/chunk"

    def slice_pair(self, images: torch.Tensor, pair_index: int):
        """
        Extract a frame pair for interpolation.

        Args:
            images: (N, H, W, C) tensor from VHS LoadVideo
            pair_index: 0-indexed pair number

        Returns:
            frame_pair: (2, H, W, C) tensor [frame_i, frame_i+1]
            pair_index: Echo back for workflow tracking
            total_pairs: N-1 (number of frame pairs)
            is_last_pair: True if pair_index == total_pairs-1

        Raises:
            ValueError: If insufficient frames or pair_index out of range
        """
        N = images.shape[0]
        total_pairs = N - 1

        # Validation
        if N < 2:
            raise ValueError(
                f"FramePairSlicer requires at least 2 frames for interpolation, got {N}. "
                f"Please load a video with multiple frames."
            )

        if pair_index >= total_pairs:
            raise ValueError(
                f"pair_index {pair_index} out of range. "
                f"Valid range: [0, {total_pairs-1}] for {N} frames. "
                f"This video has {total_pairs} frame pairs (frames 0-{N-1})."
            )

        if pair_index < 0:
            raise ValueError(
                f"pair_index must be non-negative, got {pair_index}"
            )

        # Extract pair
        frame_pair = images[pair_index:pair_index+2]  # (2, H, W, C)
        is_last_pair = (pair_index == total_pairs - 1)

        H, W = frame_pair.shape[1:3]
        print(
            f"FramePairSlicer: Extracted pair {pair_index+1}/{total_pairs} "
            f"(frames {pair_index}-{pair_index+1}) @ {H}×{W}"
        )

        return (frame_pair, pair_index, total_pairs, is_last_pair)
