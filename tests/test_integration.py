"""
Integration tests for video processing workflows.

These tests use actual test videos to verify end-to-end functionality.
Requires test videos in examples/ directory with vfi_ prefix.
"""

import pytest
import torch
from pathlib import Path

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def test_videos_dir():
    """Get path to test videos directory."""
    return Path(__file__).parent.parent / "examples"


@pytest.fixture
def test_video_files(test_videos_dir):
    """List all test video files."""
    if not test_videos_dir.exists():
        pytest.skip("examples directory not found")

    video_files = list(test_videos_dir.glob("vfi_test_*.mp4"))
    video_files.extend(test_videos_dir.glob("vfi_test_*.webm"))

    if not video_files:
        pytest.skip("No test videos found in examples/")

    return sorted(video_files)


@pytest.fixture
def smallest_test_video(test_video_files):
    """Get the smallest test video for faster testing."""
    # vfi_test_360p.mp4 is the smallest at ~1.5MB
    for video in test_video_files:
        if "360p" in video.name:
            return video
    return test_video_files[0] if test_video_files else None


class TestVideoFileAccess:
    """Test that test videos are accessible and readable."""

    def test_test_videos_exist(self, test_videos_dir):
        """Test that examples directory contains test videos."""
        video_files = list(test_videos_dir.glob("vfi_test_*"))
        assert len(video_files) > 0, "No test videos found in examples/"

    def test_video_files_readable(self, test_video_files):
        """Test that all test video files are readable."""
        for video_file in test_video_files:
            assert video_file.exists(), f"Video file not found: {video_file}"
            assert video_file.is_file(), f"Not a file: {video_file}"
            assert video_file.stat().st_size > 0, f"Empty file: {video_file}"

    def test_video_file_sizes(self, test_video_files):
        """Test that video files have expected size ranges."""
        size_ranges = {
            "360p": (1_000_000, 2_000_000),      # ~1.5MB
            "480p": (2_000_000, 3_000_000),      # ~2.4MB
            "720p": (4_000_000, 6_000_000),      # ~4.6MB
            "1080p": (6_000_000, 10_000_000),    # ~7.2MB
            "1440p": (12_000_000, 16_000_000),   # ~14.3MB
            "4K": (10_000_000, 20_000_000),      # ~13.4MB
        }

        for video_file in test_video_files:
            file_size = video_file.stat().st_size

            # Find matching resolution
            for res, (min_size, max_size) in size_ranges.items():
                if res in video_file.name:
                    assert min_size <= file_size <= max_size, (
                        f"{video_file.name}: size {file_size} not in expected range "
                        f"[{min_size}, {max_size}]"
                    )
                    break


@pytest.mark.slow
class TestVideoLoading:
    """Test video loading (requires opencv or similar)."""

    def test_can_import_cv2(self):
        """Test if opencv is available for video loading."""
        try:
            import cv2
            assert cv2 is not None
        except ImportError:
            pytest.skip("opencv-python not installed")

    @pytest.mark.skipif(True, reason="Requires opencv-python and is slow")
    def test_load_smallest_video(self, smallest_test_video):
        """Test loading the smallest test video with opencv."""
        import cv2

        cap = cv2.VideoCapture(str(smallest_test_video))
        assert cap.isOpened(), f"Failed to open video: {smallest_test_video}"

        # Read first frame
        ret, frame = cap.read()
        assert ret, "Failed to read first frame"
        assert frame is not None
        assert frame.shape[2] == 3, "Expected RGB frame"

        cap.release()


class TestFramePairSlicerWithRealDimensions:
    """Test FramePairSlicer with realistic video dimensions."""

    @pytest.mark.parametrize("resolution,height,width", [
        ("360p", 360, 640),
        ("480p", 480, 854),
        ("720p", 720, 1280),
        ("1080p", 1080, 1920),
        ("1440p", 1440, 2560),
        ("4K", 2160, 3840),
    ])
    def test_slicer_with_video_resolutions(self, resolution, height, width):
        """Test FramePairSlicer with actual video resolutions."""
        from nodes.frame_pair_slicer import FramePairSlicer

        # Create synthetic frames matching video resolution
        # ComfyUI IMAGE format: (N, H, W, C)
        frames = torch.rand(10, height, width, 3, dtype=torch.float32)

        slicer = FramePairSlicer()
        frame_pair, pair_idx, total_pairs, is_last = slicer.slice_pair(frames, pair_index=0)

        assert frame_pair.shape == (2, height, width, 3)
        assert total_pairs == 9


class TestTestVideoMetadata:
    """Test metadata about test videos."""

    def test_list_all_test_videos(self, test_video_files):
        """List all available test videos with their sizes."""
        print("\n\nAvailable test videos:")
        for video in test_video_files:
            size_mb = video.stat().st_size / 1_000_000
            print(f"  - {video.name}: {size_mb:.2f} MB")

        assert len(test_video_files) > 0

    def test_expected_resolutions_present(self, test_videos_dir):
        """Test that expected resolution test videos are present."""
        expected_resolutions = ["360p", "480p", "720p", "1080p"]

        for resolution in expected_resolutions:
            matching_files = list(test_videos_dir.glob(f"vfi_test_{resolution}.*"))
            assert len(matching_files) > 0, (
                f"Missing test video for {resolution} resolution"
            )


# Note: Actual video processing tests would require:
# 1. opencv-python or similar for video loading
# 2. Actual TLBVFI model file (~3.6GB)
# 3. GPU for reasonable performance
# 4. Much longer test execution time
#
# These are intentionally left out of CI to keep it fast and lightweight.
# For full integration testing, run locally with:
#   pytest tests/test_integration.py -v --run-slow
