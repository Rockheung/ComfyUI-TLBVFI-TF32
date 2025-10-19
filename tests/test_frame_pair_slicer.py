"""
Unit tests for FramePairSlicer node.

Tests the frame pair slicing logic without requiring ComfyUI installation.
"""

import pytest
import torch
from nodes.frame_pair_slicer import FramePairSlicer


class TestFramePairSlicerBasics:
    """Test basic functionality of FramePairSlicer."""

    def test_input_types_structure(self):
        """Test INPUT_TYPES returns correct structure."""
        input_types = FramePairSlicer.INPUT_TYPES()

        assert "required" in input_types
        assert "optional" in input_types
        assert "images" in input_types["required"]
        assert "pair_index" in input_types["optional"]

    def test_return_types(self):
        """Test RETURN_TYPES are correctly defined."""
        assert FramePairSlicer.RETURN_TYPES == ("IMAGE", "INT", "INT", "BOOLEAN")
        assert FramePairSlicer.RETURN_NAMES == ("frame_pair", "pair_index", "total_pairs", "is_last_pair")

    def test_function_name(self):
        """Test FUNCTION is correctly defined."""
        assert FramePairSlicer.FUNCTION == "slice_pair"

    def test_category(self):
        """Test node is in correct category."""
        assert FramePairSlicer.CATEGORY == "frame_interpolation/TLBVFI-TF32/chunk"


class TestFramePairSlicerSlicing:
    """Test frame pair slicing logic."""

    def test_slice_first_pair(self, sample_frames_10):
        """Test slicing the first frame pair."""
        slicer = FramePairSlicer()
        frame_pair, pair_idx, total_pairs, is_last = slicer.slice_pair(sample_frames_10, pair_index=0)

        assert frame_pair.shape == (2, 256, 256, 3), "Frame pair should have shape (2, H, W, C)"
        assert pair_idx == 0, "Should return input pair_index"
        assert total_pairs == 9, "10 frames should have 9 pairs"
        assert is_last is False, "First pair should not be last"

    def test_slice_middle_pair(self, sample_frames_10):
        """Test slicing a middle frame pair."""
        slicer = FramePairSlicer()
        frame_pair, pair_idx, total_pairs, is_last = slicer.slice_pair(sample_frames_10, pair_index=4)

        assert frame_pair.shape == (2, 256, 256, 3)
        assert pair_idx == 4
        assert total_pairs == 9
        assert is_last is False

    def test_slice_last_pair(self, sample_frames_10):
        """Test slicing the last frame pair."""
        slicer = FramePairSlicer()
        frame_pair, pair_idx, total_pairs, is_last = slicer.slice_pair(sample_frames_10, pair_index=8)

        assert frame_pair.shape == (2, 256, 256, 3)
        assert pair_idx == 8
        assert total_pairs == 9
        assert is_last is True, "Last pair should be marked as last"

    def test_correct_frames_selected(self, sample_frames_10):
        """Test that correct consecutive frames are selected."""
        slicer = FramePairSlicer()

        # Add unique values to identify frames
        for i in range(10):
            sample_frames_10[i, 0, 0, 0] = float(i)

        frame_pair, _, _, _ = slicer.slice_pair(sample_frames_10, pair_index=3)

        # Should get frames 3 and 4
        assert frame_pair[0, 0, 0, 0] == 3.0, "First frame should be frame 3"
        assert frame_pair[1, 0, 0, 0] == 4.0, "Second frame should be frame 4"


class TestFramePairSlicerEdgeCases:
    """Test edge cases and error handling."""

    def test_minimum_frames(self, sample_frames_2):
        """Test with minimum number of frames (2)."""
        slicer = FramePairSlicer()
        frame_pair, pair_idx, total_pairs, is_last = slicer.slice_pair(sample_frames_2, pair_index=0)

        assert frame_pair.shape == (2, 256, 256, 3)
        assert total_pairs == 1, "2 frames should have 1 pair"
        assert is_last is True, "Only pair should be marked as last"

    def test_insufficient_frames_error(self):
        """Test error when less than 2 frames provided."""
        slicer = FramePairSlicer()
        single_frame = torch.rand(1, 256, 256, 3)

        with pytest.raises(ValueError, match="requires at least 2 frames"):
            slicer.slice_pair(single_frame, pair_index=0)

    def test_pair_index_out_of_range_error(self, sample_frames_10):
        """Test error when pair_index is out of range."""
        slicer = FramePairSlicer()

        with pytest.raises(ValueError, match="out of range"):
            slicer.slice_pair(sample_frames_10, pair_index=9)  # 10 frames = 9 pairs, max index 8

        with pytest.raises(ValueError, match="out of range"):
            slicer.slice_pair(sample_frames_10, pair_index=100)

    def test_negative_pair_index_error(self, sample_frames_10):
        """Test error when pair_index is negative."""
        slicer = FramePairSlicer()

        with pytest.raises(ValueError, match="must be non-negative"):
            slicer.slice_pair(sample_frames_10, pair_index=-1)

    def test_large_batch(self, sample_frames_100):
        """Test with large batch of frames."""
        slicer = FramePairSlicer()
        frame_pair, _, total_pairs, _ = slicer.slice_pair(sample_frames_100, pair_index=50)

        assert frame_pair.shape == (2, 128, 128, 3)
        assert total_pairs == 99, "100 frames should have 99 pairs"


class TestFramePairSlicerDataIntegrity:
    """Test data integrity and tensor properties."""

    def test_output_is_view_not_copy(self, sample_frames_10):
        """Test that output is a view of input tensor (memory efficient)."""
        slicer = FramePairSlicer()
        original_ptr = sample_frames_10.data_ptr()

        frame_pair, _, _, _ = slicer.slice_pair(sample_frames_10, pair_index=0)

        # Frame pair should share memory with original tensor
        assert frame_pair.data_ptr() == original_ptr or frame_pair.is_contiguous()

    def test_preserves_dtype(self, sample_frames_10):
        """Test that output preserves input dtype."""
        slicer = FramePairSlicer()
        frame_pair, _, _, _ = slicer.slice_pair(sample_frames_10, pair_index=0)

        assert frame_pair.dtype == sample_frames_10.dtype

    def test_preserves_device(self, sample_frames_10):
        """Test that output preserves input device."""
        slicer = FramePairSlicer()
        frame_pair, _, _, _ = slicer.slice_pair(sample_frames_10, pair_index=0)

        assert frame_pair.device == sample_frames_10.device

    @pytest.mark.requires_gpu
    def test_works_with_gpu_tensors(self, sample_frames_10, skip_if_no_gpu):
        """Test slicing works with GPU tensors."""
        gpu_frames = sample_frames_10.cuda()
        slicer = FramePairSlicer()

        frame_pair, _, _, _ = slicer.slice_pair(gpu_frames, pair_index=0)

        assert frame_pair.is_cuda
        assert frame_pair.device == gpu_frames.device


class TestFramePairSlicerResolutions:
    """Test with various video resolutions."""

    @pytest.mark.parametrize("height,width", [
        (256, 256),    # Square
        (720, 1280),   # HD
        (1080, 1920),  # Full HD
        (2160, 3840),  # 4K
        (512, 768),    # Arbitrary
    ])
    def test_various_resolutions(self, height, width):
        """Test slicing works with various resolutions."""
        frames = torch.rand(5, height, width, 3)
        slicer = FramePairSlicer()

        frame_pair, _, _, _ = slicer.slice_pair(frames, pair_index=2)

        assert frame_pair.shape == (2, height, width, 3)


class TestFramePairSlicerSequential:
    """Test sequential slicing scenarios."""

    def test_sequential_slicing_covers_all_frames(self, sample_frames_10):
        """Test that sequential slicing covers all frame pairs."""
        slicer = FramePairSlicer()
        total_frames = sample_frames_10.shape[0]

        # Mark each frame with unique value
        for i in range(total_frames):
            sample_frames_10[i, 0, 0, 0] = float(i)

        # Slice all pairs sequentially
        for pair_idx in range(total_frames - 1):
            frame_pair, returned_idx, total_pairs, is_last = slicer.slice_pair(
                sample_frames_10, pair_index=pair_idx
            )

            # Verify correct frames
            assert frame_pair[0, 0, 0, 0] == float(pair_idx)
            assert frame_pair[1, 0, 0, 0] == float(pair_idx + 1)

            # Verify metadata
            assert returned_idx == pair_idx
            assert total_pairs == total_frames - 1
            assert is_last == (pair_idx == total_frames - 2)

    def test_non_sequential_slicing(self, sample_frames_10):
        """Test that non-sequential slicing works correctly."""
        slicer = FramePairSlicer()

        # Slice in random order
        indices = [5, 2, 8, 0, 3]

        for idx in indices:
            frame_pair, returned_idx, _, _ = slicer.slice_pair(sample_frames_10, pair_index=idx)
            assert returned_idx == idx
            assert frame_pair.shape == (2, 256, 256, 3)
