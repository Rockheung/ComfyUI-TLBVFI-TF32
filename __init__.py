# Legacy monolithic node (backward compatibility)
from .tlbvfi_node import TLBVFI_VFI_TF32

# Chunk-based workflow nodes
from .nodes import (
    FramePairSlicer,
    TLBVFI_FrameFromBatch,
    TLBVFI_Interpolator,
    TLBVFI_Interpolator_V2,
    TLBVFI_BatchInterpolator_V2,
    ChunkVideoSaver,
    VideoConcatenator,
    TLBVFI_ChunkProcessor,
    TLBVFI_ClearModelCache,
)

NODE_CLASS_MAPPINGS = {
    # Legacy node (for backward compatibility)
    "TLBVFI_VFI_TF32": TLBVFI_VFI_TF32,

    # Chunk-based workflow nodes
    "TLBVFI_FramePairSlicer": FramePairSlicer,
    "TLBVFI_FrameFromBatch": TLBVFI_FrameFromBatch,
    "TLBVFI_Interpolator": TLBVFI_Interpolator,
    "TLBVFI_Interpolator_V2": TLBVFI_Interpolator_V2,
    "TLBVFI_BatchInterpolator_V2": TLBVFI_BatchInterpolator_V2,
    "TLBVFI_ChunkVideoSaver": ChunkVideoSaver,
    "TLBVFI_VideoConcatenator": VideoConcatenator,
    "TLBVFI_ChunkProcessor": TLBVFI_ChunkProcessor,
    "TLBVFI_ClearModelCache": TLBVFI_ClearModelCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Legacy
    "TLBVFI_VFI_TF32": "TLBVFI Frame Interpolation (TF32) [Legacy]",

    # Chunk-based workflow
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

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
