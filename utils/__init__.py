# ComfyUI-TLBVFI-TF32 Shared Utilities
#
# This module contains shared utility functions for model loading,
# memory management, and manifest operations.

from .model_loader import (
    load_tlbvfi_model,
    enable_tf32_if_available,
    enable_cudnn_benchmark,
    dict2namespace,
)

from .memory_manager import (
    cleanup_memory,
    get_memory_stats,
    check_memory_available,
    estimate_frame_memory,
    print_memory_summary,
)

from .manifest_manager import (
    create_session_id,
    create_manifest,
    load_manifest,
    save_manifest,
    add_chunk_to_manifest,
    get_chunk_paths,
    get_session_stats,
    cleanup_session,
)

__all__ = [
    # Model loading
    'load_tlbvfi_model',
    'enable_tf32_if_available',
    'enable_cudnn_benchmark',
    'dict2namespace',
    # Memory management
    'cleanup_memory',
    'get_memory_stats',
    'check_memory_available',
    'estimate_frame_memory',
    'print_memory_summary',
    # Manifest management
    'create_session_id',
    'create_manifest',
    'load_manifest',
    'save_manifest',
    'add_chunk_to_manifest',
    'get_chunk_paths',
    'get_session_stats',
    'cleanup_session',
]
