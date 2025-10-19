"""
Test actual 4K video frame interpolation on GPU.

This test requires:
- CUDA GPU (RTX 4090)
- Model file: vimeo_unet.pth in ComfyUI/models/interpolation/
- Test video: vfi_test_4K.webm in examples/

Run with:
    pytest tests/test_4k_interpolation.py -v -s
    # or
    uv run pytest tests/test_4k_interpolation.py -v -s
"""

import pytest
import torch
from pathlib import Path


pytestmark = [
    pytest.mark.requires_gpu,
    pytest.mark.requires_model,
    pytest.mark.slow,
]


@pytest.fixture
def test_4k_video():
    """Get 4K test video path."""
    video_path = Path(__file__).parent.parent / "examples" / "vfi_test_4K.webm"
    if not video_path.exists():
        pytest.skip(f"4K test video not found: {video_path}")
    return video_path


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if no GPU available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available")

    print(f"\n✅ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")


class Test4KInterpolation:
    """Test real 4K video frame interpolation."""

    def test_load_4k_video_frames(self, test_4k_video, skip_if_no_gpu):
        """Test loading 4K video frames with OpenCV."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("opencv-python not installed")

        # Load video
        cap = cv2.VideoCapture(str(test_4k_video))
        assert cap.isOpened(), f"Failed to open {test_4k_video}"

        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n📹 Video Info:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {frame_count}")

        assert width == 3840, f"Expected 4K width 3840, got {width}"
        assert height == 2160, f"Expected 4K height 2160, got {height}"

        # Load first 2 frames for interpolation test
        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()
        cap.release()

        assert ret1 and ret2, "Failed to read first 2 frames"
        assert frame1.shape == (2160, 3840, 3)
        assert frame2.shape == (2160, 3840, 3)

        print(f"✅ Loaded 2 frames successfully")

    def test_interpolate_4k_frames(self, test_4k_video, skip_if_no_gpu):
        """Test actual 4K frame interpolation with TLBVFI model."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("opencv-python not installed")

        from nodes.tlbvfi_interpolator_v2 import TLBVFI_Interpolator_V2

        # Load first 2 frames
        cap = cv2.VideoCapture(str(test_4k_video))
        ret1, frame1_bgr = cap.read()
        ret2, frame2_bgr = cap.read()
        cap.release()

        assert ret1 and ret2

        # Convert to ComfyUI format: BGR uint8 -> RGB float32
        frame1_rgb = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frame2_rgb = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Add batch dimension: (H,W,C) -> (1,H,W,C)
        frame1_batch = torch.from_numpy(np.expand_dims(frame1_rgb, axis=0))
        frame2_batch = torch.from_numpy(np.expand_dims(frame2_rgb, axis=0))

        print(f"\n🖼️  Input frames shape: {frame1_batch.shape}")

        # Initialize interpolator
        interpolator = TLBVFI_Interpolator_V2()

        # Run interpolation (1x = 1 intermediate frame)
        print(f"🚀 Starting 4K frame interpolation...")
        print(f"   This may take 30-60 seconds on RTX 4090...")

        try:
            result = interpolator.interpolate(
                prev_frame=frame1_batch,
                next_frame=frame2_batch,
                times_to_interpolate=1,  # 1x = 1 intermediate frame
                model_name="vimeo_unet.pth",
                sample_steps=10,
                enable_tf32=True,  # RTX 30/40 optimization
                flow_scale=0.5,
                cpu_offload=False,  # Keep on GPU
                gpu_id=0
            )

            # Result is a tuple: (interpolated_frames,)
            interpolated_frames = result[0]

            print(f"\n✅ Interpolation successful!")
            print(f"   Output shape: {interpolated_frames.shape}")
            print(f"   Expected: (3, 2160, 3840, 3)")  # 3 frames: prev, interpolated, next

            # Verify output
            assert interpolated_frames.shape == (3, 2160, 3840, 3), \
                f"Expected (3, 2160, 3840, 3), got {interpolated_frames.shape}"
            assert interpolated_frames.dtype == torch.float32
            assert 0.0 <= interpolated_frames.min() <= 1.0
            assert 0.0 <= interpolated_frames.max() <= 1.0

            # Check GPU memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                print(f"\n💾 GPU Memory:")
                print(f"   Allocated: {memory_allocated:.2f} GB")
                print(f"   Reserved: {memory_reserved:.2f} GB")

            print(f"\n🎉 4K interpolation test PASSED!")

        except FileNotFoundError as e:
            if "vimeo_unet.pth" in str(e):
                pytest.skip("Model file vimeo_unet.pth not found. Download from https://huggingface.co/ucfzl/TLBVFI/tree/main")
            raise

        except Exception as e:
            print(f"\n❌ Interpolation failed: {e}")
            raise


class Test4KPerformance:
    """Test 4K interpolation performance metrics."""

    def test_4k_interpolation_speed(self, test_4k_video, skip_if_no_gpu):
        """Measure 4K interpolation performance on RTX 4090."""
        import time
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("opencv-python not installed")

        from nodes.tlbvfi_interpolator_v2 import TLBVFI_Interpolator_V2

        # Load frames
        cap = cv2.VideoCapture(str(test_4k_video))
        ret1, frame1_bgr = cap.read()
        ret2, frame2_bgr = cap.read()
        cap.release()

        # Convert to ComfyUI format
        frame1_rgb = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frame2_rgb = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frame1_batch = torch.from_numpy(np.expand_dims(frame1_rgb, axis=0))
        frame2_batch = torch.from_numpy(np.expand_dims(frame2_rgb, axis=0))

        interpolator = TLBVFI_Interpolator_V2()

        # Warmup run
        print(f"\n🔥 Warmup run...")
        try:
            _ = interpolator.interpolate(
                frame1_batch, frame2_batch, times_to_interpolate=1,
                model_name="vimeo_unet.pth", sample_steps=10,
                enable_tf32=True, flow_scale=0.5, cpu_offload=False, gpu_id=0
            )
        except FileNotFoundError:
            pytest.skip("Model file not found")

        # Timed run
        print(f"\n⏱️  Performance test...")
        start_time = time.time()

        result = interpolator.interpolate(
            frame1_batch, frame2_batch, times_to_interpolate=1,
            model_name="vimeo_unet.pth", sample_steps=10,
            enable_tf32=True, flow_scale=0.5, cpu_offload=False, gpu_id=0
        )

        elapsed_time = time.time() - start_time

        print(f"\n📊 Performance Results:")
        print(f"   Resolution: 4K (3840x2160)")
        print(f"   Time: {elapsed_time:.2f} seconds")
        print(f"   FPS: {1.0/elapsed_time:.2f}")

        # RTX 4090 should handle 4K in reasonable time
        # This is informational, not a hard requirement
        if elapsed_time < 60:
            print(f"   ✅ Good performance (< 60s)")
        else:
            print(f"   ⚠️  Slow performance (> 60s)")
