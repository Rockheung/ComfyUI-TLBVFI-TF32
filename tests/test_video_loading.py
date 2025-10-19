"""
Test actual video loading and frame extraction.

Tests real video file loading and conversion to ComfyUI IMAGE format.
Requires opencv-python installed.
"""

import pytest
import torch
from pathlib import Path

# Mark all tests as integration
pytestmark = pytest.mark.integration


@pytest.fixture
def test_videos_dir():
    """Get path to test videos directory."""
    return Path(__file__).parent.parent / "examples"


@pytest.fixture
def smallest_test_video(test_videos_dir):
    """Get the smallest test video for faster testing."""
    video_file = test_videos_dir / "vfi_test_360p.mp4"
    if not video_file.exists():
        pytest.skip("Test video not found: vfi_test_360p.mp4")
    return video_file


class TestOpenCVVideoLoading:
    """Test video loading with OpenCV."""

    def test_opencv_available(self):
        """Test that opencv is installed."""
        try:
            import cv2
            assert cv2.__version__ is not None
            print(f"\nOpenCV version: {cv2.__version__}")
        except ImportError:
            pytest.skip("opencv-python not installed")

    def test_load_video_basic(self, smallest_test_video):
        """Test basic video loading."""
        try:
            import cv2
        except ImportError:
            pytest.skip("opencv-python not installed")

        cap = cv2.VideoCapture(str(smallest_test_video))
        assert cap.isOpened(), f"Failed to open video: {smallest_test_video}"

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\nVideo properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Frame count: {frame_count}")

        assert fps > 0, "Invalid FPS"
        assert frame_count > 0, "No frames in video"
        assert width == 640, f"Expected width 640, got {width}"
        assert height == 360, f"Expected height 360, got {height}"

        cap.release()

    def test_read_first_frame(self, smallest_test_video):
        """Test reading the first frame."""
        try:
            import cv2
        except ImportError:
            pytest.skip("opencv-python not installed")

        cap = cv2.VideoCapture(str(smallest_test_video))
        assert cap.isOpened()

        ret, frame = cap.read()
        assert ret, "Failed to read first frame"
        assert frame is not None
        assert frame.shape == (360, 640, 3), f"Unexpected frame shape: {frame.shape}"
        assert frame.dtype == 'uint8', f"Unexpected dtype: {frame.dtype}"

        cap.release()

    def test_read_all_frames(self, smallest_test_video):
        """Test reading all frames from video."""
        try:
            import cv2
        except ImportError:
            pytest.skip("opencv-python not installed")

        cap = cv2.VideoCapture(str(smallest_test_video))
        frames_read = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_read += 1

        cap.release()

        print(f"\nTotal frames read: {frames_read}")
        assert frames_read > 0, "No frames read from video"
        # ~8 seconds at 30fps = ~240 frames
        assert 200 < frames_read < 300, f"Unexpected frame count: {frames_read}"


class TestVideoToComfyUIFormat:
    """Test converting video frames to ComfyUI IMAGE format."""

    def test_convert_frame_to_comfyui_format(self, smallest_test_video):
        """Test converting OpenCV frame to ComfyUI format."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("opencv-python not installed")

        cap = cv2.VideoCapture(str(smallest_test_video))
        ret, frame = cap.read()
        cap.release()

        assert ret and frame is not None

        # OpenCV frame: (H, W, C) in BGR, uint8 [0, 255]
        # ComfyUI IMAGE: (N, H, W, C) in RGB, float32 [0, 1]

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to float32 and normalize
        frame_float = frame_rgb.astype(np.float32) / 255.0

        # Add batch dimension
        frame_batch = np.expand_dims(frame_float, axis=0)

        # Convert to torch tensor
        frame_tensor = torch.from_numpy(frame_batch)

        # Verify ComfyUI format
        assert frame_tensor.shape == (1, 360, 640, 3), f"Wrong shape: {frame_tensor.shape}"
        assert frame_tensor.dtype == torch.float32, f"Wrong dtype: {frame_tensor.dtype}"
        assert 0.0 <= frame_tensor.min() <= 1.0, f"Values out of range: min={frame_tensor.min()}"
        assert 0.0 <= frame_tensor.max() <= 1.0, f"Values out of range: max={frame_tensor.max()}"

        print(f"\nConverted frame shape: {frame_tensor.shape}")
        print(f"Value range: [{frame_tensor.min():.3f}, {frame_tensor.max():.3f}]")

    def test_load_video_as_batch(self, smallest_test_video):
        """Test loading entire video as ComfyUI IMAGE batch."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("opencv-python not installed")

        cap = cv2.VideoCapture(str(smallest_test_video))

        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to ComfyUI format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_float = frame_rgb.astype(np.float32) / 255.0
            frames.append(frame_float)

        cap.release()

        # Stack into batch
        frames_array = np.stack(frames, axis=0)
        frames_tensor = torch.from_numpy(frames_array)

        # Verify batch format
        N, H, W, C = frames_tensor.shape
        print(f"\nLoaded video batch:")
        print(f"  Shape: {frames_tensor.shape}")
        print(f"  Frames: {N}")
        print(f"  Resolution: {W}x{H}")
        print(f"  Channels: {C}")

        assert frames_tensor.shape[1:] == (360, 640, 3)
        assert frames_tensor.dtype == torch.float32
        assert 200 < N < 300, f"Unexpected number of frames: {N}"

    def test_extract_frame_pairs(self, smallest_test_video):
        """Test extracting frame pairs from video."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("opencv-python not installed")

        from nodes.frame_pair_slicer import FramePairSlicer

        # Load video
        cap = cv2.VideoCapture(str(smallest_test_video))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_float = frame_rgb.astype(np.float32) / 255.0
            frames.append(frame_float)
        cap.release()

        # Convert to tensor
        frames_tensor = torch.from_numpy(np.stack(frames[:10], axis=0))  # Use first 10 frames
        print(f"\nLoaded {frames_tensor.shape[0]} frames for pair extraction")

        # Test FramePairSlicer
        slicer = FramePairSlicer()
        frame_pair, pair_idx, total_pairs, is_last = slicer.slice_pair(frames_tensor, pair_index=0)

        assert frame_pair.shape == (2, 360, 640, 3)
        assert total_pairs == 9
        assert pair_idx == 0
        assert is_last is False

        print(f"  Extracted pair shape: {frame_pair.shape}")
        print(f"  Total pairs available: {total_pairs}")


class TestMultipleResolutions:
    """Test loading videos of different resolutions."""

    @pytest.mark.parametrize("video_name,expected_resolution", [
        ("vfi_test_360p.mp4", (360, 640)),
        ("vfi_test_480p.mp4", (480, 854)),
        ("vfi_test_720p.mp4", (720, 1280)),
        ("vfi_test_1080p.mp4", (1080, 1920)),
        ("vfi_test_4K.webm", (2160, 3840)),
    ])
    def test_load_different_resolutions(self, test_videos_dir, video_name, expected_resolution):
        """Test loading videos of different resolutions."""
        try:
            import cv2
        except ImportError:
            pytest.skip("opencv-python not installed")

        video_file = test_videos_dir / video_name
        if not video_file.exists():
            pytest.skip(f"Video not found: {video_name}")

        cap = cv2.VideoCapture(str(video_file))
        assert cap.isOpened(), f"Failed to open {video_name}"

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()

        expected_height, expected_width = expected_resolution
        assert height == expected_height, f"{video_name}: expected height {expected_height}, got {height}"
        assert width == expected_width, f"{video_name}: expected width {expected_width}, got {width}"

        print(f"\n{video_name}: {width}x{height} ✓")


class TestVideoQuality:
    """Test video quality and characteristics."""

    def test_video_not_corrupted(self, smallest_test_video):
        """Test that video is not corrupted."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("opencv-python not installed")

        cap = cv2.VideoCapture(str(smallest_test_video))

        # Read several frames and check they're valid
        for i in range(10):
            ret, frame = cap.read()
            assert ret, f"Failed to read frame {i}"
            assert frame is not None
            assert not np.all(frame == 0), f"Frame {i} is all black"
            assert not np.all(frame == 255), f"Frame {i} is all white"

        cap.release()

    def test_video_has_variation(self, smallest_test_video):
        """Test that video has visual variation (not static)."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("opencv-python not installed")

        cap = cv2.VideoCapture(str(smallest_test_video))

        ret, frame1 = cap.read()
        assert ret

        # Read frame from middle of video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)

        ret, frame2 = cap.read()
        assert ret

        cap.release()

        # Frames should be different
        difference = np.mean(np.abs(frame1.astype(float) - frame2.astype(float)))
        print(f"\nFrame difference (mean absolute): {difference:.2f}")

        # If difference is very small, video might be static
        assert difference > 1.0, "Video appears to be static (frames too similar)"
