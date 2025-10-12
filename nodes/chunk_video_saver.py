"""
ChunkVideoSaver Node for TLBVFI Chunk-Based Workflow

Saves interpolated frame chunks to disk with metadata tracking.
Enables memory-efficient processing by immediately persisting results.
"""

import torch
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    create_session_id,
    add_chunk_to_manifest,
)

import folder_paths


class ChunkVideoSaver:
    """
    Save interpolated frames to disk as chunks with manifest tracking.

    This node immediately persists frames to disk, freeing memory for
    processing subsequent chunks. Chunks are tracked in a manifest for
    later concatenation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),  # Interpolated frames from TLBVFI_Interpolator
                "chunk_id": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "session_id": ("STRING", {"default": ""}),  # Auto-generate if empty
                "output_dir": ("STRING", {"default": ""}),  # Use ComfyUI default if empty
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("session_id", "chunk_path", "num_frames")
    FUNCTION = "save_chunk"
    CATEGORY = "frame_interpolation/TLBVFI-TF32/chunk"

    DESCRIPTION = """
Save interpolated frames to disk as chunks with manifest tracking.

📌 Purpose:
- Immediately persists frames to disk, freeing memory
- Tracks chunks in JSON manifest for later concatenation
- Enables processing unlimited video length

🎯 Usage:
1. Connect interpolated_frames from TLBVFI_Interpolator
2. Set chunk_id (0, 1, 2, ...) - increment for each chunk
3. Leave session_id empty for auto-generation (first chunk)
4. Use same session_id for all chunks in same video

💾 Storage:
- Format: PyTorch .pt files (fast I/O, FP32 precision)
- Location: output_dir/tlbvfi_chunks/session_id/
- Manifest: JSON metadata for resumable processing
- Cleanup: Chunks auto-deleted after VideoConcatenator

📊 Disk space: ~1.5GB per chunk (15 frames @ 4K)
    """

    def save_chunk(self, frames: torch.Tensor, chunk_id: int, session_id: str = "",
                   output_dir: str = ""):
        """
        Save frames to disk as a chunk file.

        Args:
            frames: (N, H, W, C) tensor from TLBVFI_Interpolator
            chunk_id: Sequential chunk number
            session_id: Unique session identifier (auto-generated if empty)
            output_dir: Output directory (use default if empty)

        Returns:
            session_id: Echo back for workflow tracking
            chunk_path: Absolute path to saved chunk
            num_frames: Number of frames in chunk
        """
        # Generate session_id if not provided
        if not session_id:
            session_id = create_session_id()
            print(f"ChunkVideoSaver: Created new session {session_id}")

        # Determine output directory
        if not output_dir:
            output_dir = folder_paths.get_output_directory()

        session_dir = os.path.join(output_dir, "tlbvfi_chunks", session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Save chunk as .pt file (preserves FP32 precision, fast I/O)
        chunk_filename = f"chunk_{chunk_id:06d}.pt"
        chunk_path = os.path.join(session_dir, chunk_filename)

        # Convert to CPU if needed
        frames_cpu = frames.cpu() if frames.device.type != 'cpu' else frames

        # Save with metadata
        chunk_data = {
            'frames': frames_cpu,
            'chunk_id': chunk_id,
            'shape': frames_cpu.shape,
            'dtype': str(frames_cpu.dtype),
        }

        # Atomic write: write to temp file then rename
        temp_path = f"{chunk_path}.tmp"
        torch.save(chunk_data, temp_path)
        os.replace(temp_path, chunk_path)

        # Update manifest
        add_chunk_to_manifest(
            session_dir=session_dir,
            chunk_id=chunk_id,
            chunk_path=chunk_path,
            shape=frames_cpu.shape,
            status='complete'
        )

        num_frames = frames_cpu.shape[0]
        chunk_size_mb = os.path.getsize(chunk_path) / (1024**2)
        H, W = frames_cpu.shape[1:3]

        print(
            f"ChunkVideoSaver: Saved chunk {chunk_id} → {chunk_path}\n"
            f"  {num_frames} frames @ {H}×{W}, {chunk_size_mb:.1f}MB"
        )

        return (session_id, chunk_path, num_frames)
