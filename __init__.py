# Legacy monolithic node (backward compatibility)
from .tlbvfi_node import TLBVFI_VFI_TF32

# New chunk-based workflow nodes
from .nodes import (
    FramePairSlicer,
    TLBVFI_Interpolator,
    ChunkVideoSaver,
    VideoConcatenator,
    ChunkVideoSaverV2,
    VideoConcatenatorV2,
)

NODE_CLASS_MAPPINGS = {
    # Legacy node (for backward compatibility)
    "TLBVFI_VFI_TF32": TLBVFI_VFI_TF32,

    # Chunk-based workflow nodes (raw .pt storage)
    "TLBVFI_FramePairSlicer": FramePairSlicer,
    "TLBVFI_Interpolator": TLBVFI_Interpolator,
    "TLBVFI_ChunkVideoSaver": ChunkVideoSaver,
    "TLBVFI_VideoConcatenator": VideoConcatenator,

    # Chunk-based workflow nodes (video-encoded storage - RECOMMENDED)
    "TLBVFI_ChunkVideoSaverV2": ChunkVideoSaverV2,
    "TLBVFI_VideoConcatenatorV2": VideoConcatenatorV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Legacy
    "TLBVFI_VFI_TF32": "TLBVFI Frame Interpolation (TF32) [Legacy]",

    # Chunk-based workflow (raw .pt storage)
    "TLBVFI_FramePairSlicer": "TLBVFI Frame Pair Slicer",
    "TLBVFI_Interpolator": "TLBVFI Interpolator (Chunk Mode)",
    "TLBVFI_ChunkVideoSaver": "TLBVFI Chunk Saver (Raw .pt)",
    "TLBVFI_VideoConcatenator": "TLBVFI Video Concatenator (Raw .pt)",

    # Chunk-based workflow (video-encoded storage - RECOMMENDED)
    "TLBVFI_ChunkVideoSaverV2": "TLBVFI Chunk Saver V2 (Video Encoded)",
    "TLBVFI_VideoConcatenatorV2": "TLBVFI Video Concatenator V2 (FFmpeg)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']