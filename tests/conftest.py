"""
Pytest configuration and fixtures for ComfyUI-TLBVFI-TF32 tests.

Provides common test fixtures and utilities for testing custom nodes
without requiring ComfyUI installation.
"""

import os
import pytest
import torch
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Generator

# Set PyTorch CUDA memory allocator configuration BEFORE any CUDA operations
# This helps prevent memory fragmentation on large allocations (e.g., 4K video)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock folder_paths BEFORE any imports
mock_folder_paths = MagicMock()
mock_folder_paths.get_folder_paths = MagicMock(return_value=[str(PROJECT_ROOT / "models")])
# Mock get_filename_list to return a fake model file so INPUT_TYPES() doesn't fail
mock_folder_paths.get_filename_list = MagicMock(return_value=["vimeo_unet.pth"])

# Mock get_full_path to return actual ComfyUI model path
def mock_get_full_path(folder_name, filename):
    """Return full path to model file in ComfyUI installation."""
    if folder_name == "interpolation":
        # Try ComfyUI installation directory
        comfyui_model_path = Path("D:/Users/rock/Documents/ComfyUI/models/interpolation") / filename
        if comfyui_model_path.exists():
            return str(comfyui_model_path)
    return None

mock_folder_paths.get_full_path = mock_get_full_path
sys.modules['folder_paths'] = mock_folder_paths

# Mock comfy modules BEFORE any imports
mock_model_management = MagicMock()
mock_model_management.soft_empty_cache = MagicMock()
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.model_management'] = mock_model_management

mock_comfy_utils = MagicMock()
mock_comfy_utils.should_stop_processing = MagicMock(return_value=False)
sys.modules['comfy.utils'] = mock_comfy_utils

sys.modules['execution'] = MagicMock()


# Mock ComfyUI dependencies
@pytest.fixture(scope="session", autouse=True)
def mock_comfy_dependencies() -> Generator:
    """Mock ComfyUI dependencies so tests can run without ComfyUI installation."""
    # Mocks are already set up at module level, just yield

    # Additional patching for chunk_processor's and tlbvfi_interpolator's find_models functions
    # This prevents them from checking for actual model files
    import nodes.chunk_processor
    import nodes.tlbvfi_interpolator

    original_chunk_find_models = getattr(nodes.chunk_processor, 'find_models', None)
    original_interpolator_find_models = getattr(nodes.tlbvfi_interpolator, 'find_models', None)

    if original_chunk_find_models:
        # Mock find_models to always return a dummy model
        nodes.chunk_processor.find_models = lambda folder_type, extensions: ["vimeo_unet.pth"]

    if original_interpolator_find_models:
        nodes.tlbvfi_interpolator.find_models = lambda folder_type, extensions: ["vimeo_unet.pth"]

    yield

    # Restore original functions if they existed
    if original_chunk_find_models:
        nodes.chunk_processor.find_models = original_chunk_find_models
    if original_interpolator_find_models:
        nodes.tlbvfi_interpolator.find_models = original_interpolator_find_models


@pytest.fixture
def sample_frames_2() -> torch.Tensor:
    """Generate 2 sample frames for testing (ComfyUI IMAGE format: N, H, W, C)."""
    # Create 2 frames of 256x256 RGB images
    frames = torch.rand(2, 256, 256, 3, dtype=torch.float32)
    return frames


@pytest.fixture
def sample_frames_10() -> torch.Tensor:
    """Generate 10 sample frames for testing."""
    frames = torch.rand(10, 256, 256, 3, dtype=torch.float32)
    return frames


@pytest.fixture
def sample_frames_100() -> torch.Tensor:
    """Generate 100 sample frames for stress testing."""
    frames = torch.rand(100, 128, 128, 3, dtype=torch.float32)
    return frames


@pytest.fixture
def sample_4k_frame() -> torch.Tensor:
    """Generate a single 4K frame for memory testing."""
    frame = torch.rand(1, 2160, 3840, 3, dtype=torch.float32)
    return frame


@pytest.fixture
def sample_frame_pair() -> torch.Tensor:
    """Generate a frame pair for interpolation testing."""
    pair = torch.rand(2, 256, 256, 3, dtype=torch.float32)
    return pair


@pytest.fixture
def mock_tlbvfi_model():
    """Mock TLBVFI model for testing without actual model file."""
    mock_model = MagicMock()
    mock_model.sample = MagicMock(return_value=torch.rand(1, 256, 256, 3))
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.eval = MagicMock(return_value=mock_model)
    return mock_model


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Create temporary output directory for test artifacts."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def temp_chunk_dir(tmp_path) -> Path:
    """Create temporary directory for chunk storage."""
    chunk_dir = tmp_path / "tlbvfi_chunks"
    chunk_dir.mkdir()
    return chunk_dir


@pytest.fixture(autouse=True)
def reset_model_cache():
    """Reset model cache before each test to ensure isolation."""
    # Import here to avoid circular dependencies
    try:
        from nodes.tlbvfi_interpolator_v2 import clear_model_cache
        clear_model_cache()
    except ImportError:
        pass

    yield

    # Cleanup after test
    try:
        from nodes.tlbvfi_interpolator_v2 import clear_model_cache
        clear_model_cache()
    except ImportError:
        pass


@pytest.fixture
def gpu_available() -> bool:
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture
def skip_if_no_gpu(gpu_available):
    """Skip test if GPU is not available."""
    if not gpu_available:
        pytest.skip("GPU not available")


# Utility functions for tests
def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple, msg: str = ""):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, (
        f"{msg}Expected shape {expected_shape}, got {tensor.shape}"
    )


def assert_tensor_range(tensor: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0):
    """Assert tensor values are within expected range."""
    assert tensor.min() >= min_val, f"Tensor min {tensor.min()} < {min_val}"
    assert tensor.max() <= max_val, f"Tensor max {tensor.max()} > {max_val}"


# Make utilities available to tests
pytest.assert_tensor_shape = assert_tensor_shape
pytest.assert_tensor_range = assert_tensor_range


@pytest.fixture(scope="session")
def node_mappings():
    """
    Provide NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS for testing.

    This reconstructs the mappings from actual node imports instead of
    importing from __init__.py which uses relative imports.
    """
    # Import nodes directly from their modules
    from nodes.frame_pair_slicer import FramePairSlicer
    from nodes.tlbvfi_interpolator_v2 import TLBVFI_Interpolator_V2
    from nodes.batch_interpolator_v2 import TLBVFI_BatchInterpolator_V2
    from nodes.model_cache_manager import TLBVFI_ClearModelCache
    from nodes.chunk_processor import TLBVFI_ChunkProcessor

    # Try to import other nodes (some might not be available in tests)
    try:
        from tlbvfi_node import TLBVFI_VFI_TF32
    except ImportError:
        TLBVFI_VFI_TF32 = None

    try:
        from nodes.frame_from_batch import TLBVFI_FrameFromBatch
    except ImportError:
        TLBVFI_FrameFromBatch = None

    try:
        from nodes.tlbvfi_interpolator import TLBVFI_Interpolator
    except ImportError:
        TLBVFI_Interpolator = None

    try:
        from nodes.chunk_video_saver import ChunkVideoSaver
    except ImportError:
        ChunkVideoSaver = None

    try:
        from nodes.video_concatenator import VideoConcatenator
    except ImportError:
        VideoConcatenator = None

    # Build mappings (matching __init__.py structure)
    node_class_mappings = {
        "TLBVFI_FramePairSlicer": FramePairSlicer,
        "TLBVFI_Interpolator_V2": TLBVFI_Interpolator_V2,
        "TLBVFI_BatchInterpolator_V2": TLBVFI_BatchInterpolator_V2,
        "TLBVFI_ChunkProcessor": TLBVFI_ChunkProcessor,
        "TLBVFI_ClearModelCache": TLBVFI_ClearModelCache,
    }

    # Add optional nodes if available
    if TLBVFI_VFI_TF32:
        node_class_mappings["TLBVFI_VFI_TF32"] = TLBVFI_VFI_TF32
    if TLBVFI_FrameFromBatch:
        node_class_mappings["TLBVFI_FrameFromBatch"] = TLBVFI_FrameFromBatch
    if TLBVFI_Interpolator:
        node_class_mappings["TLBVFI_Interpolator"] = TLBVFI_Interpolator
    if ChunkVideoSaver:
        node_class_mappings["TLBVFI_ChunkVideoSaver"] = ChunkVideoSaver
    if VideoConcatenator:
        node_class_mappings["TLBVFI_VideoConcatenator"] = VideoConcatenator

    node_display_name_mappings = {
        "TLBVFI_VFI_TF32": "TLBVFI Frame Interpolation (TF32) [Legacy]",
        "TLBVFI_FramePairSlicer": "TLBVFI Frame Pair Slicer",
        "TLBVFI_FrameFromBatch": "TLBVFI Frame From Batch",
        "TLBVFI_Interpolator": "TLBVFI Interpolator [V1]",
        "TLBVFI_Interpolator_V2": "TLBVFI Interpolator V2 [Production]",
        "TLBVFI_BatchInterpolator_V2": "TLBVFI Batch Interpolator V2",
        "TLBVFI_ChunkVideoSaver": "TLBVFI Chunk Video Saver",
        "TLBVFI_VideoConcatenator": "TLBVFI Video Concatenator",
        "TLBVFI_ChunkProcessor": "TLBVFI Chunk Processor (All-in-One)",
        "TLBVFI_ClearModelCache": "TLBVFI Clear Model Cache",
    }

    return {
        "NODE_CLASS_MAPPINGS": node_class_mappings,
        "NODE_DISPLAY_NAME_MAPPINGS": node_display_name_mappings
    }
