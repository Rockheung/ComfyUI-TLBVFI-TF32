"""
VideoConcatenator Node for TLBVFI Chunk-Based Workflow

Loads and concatenates all saved chunks into final video tensor.
Final step in chunk-based workflow that assembles complete interpolated video.
"""

import torch
import os
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    load_manifest,
    get_chunk_paths,
    get_session_stats,
    cleanup_session,
)

import folder_paths


class VideoConcatenator:
    """
    Load and concatenate all chunks into final video.

    This node reads all saved chunks from a session, concatenates them
    into a single tensor, and optionally cleans up chunk files.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_id": ("STRING", {"default": ""}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": ""}),  # Match ChunkVideoSaver
                "cleanup_chunks": ("BOOLEAN", {"default": True}),  # Delete chunks after concat
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("video", "total_frames", "stats")
    FUNCTION = "concatenate"
    CATEGORY = "frame_interpolation/TLBVFI-TF32/chunk"

    DESCRIPTION = """
Load and concatenate all saved chunks into final video.

📌 Purpose:
- Assembles all chunks into complete interpolated video
- Final step in chunk-based workflow
- Validates chunk consistency and handles overlaps

🎯 Usage:
1. Enter session_id from ChunkVideoSaver
2. Set cleanup_chunks=True to auto-delete chunks after merge
3. Output: Complete video tensor ready for VHS SaveVideo

⚙️ Features:
- Automatic chunk ordering via manifest
- Handles overlapping frames (skips duplicates)
- Shape validation prevents concatenation errors
- Detailed error messages for debugging

💾 Memory: Peak = total frames in final video
- 1 min 1080p: ~3GB
- 10 min 4K: ~26GB (still much better than 13TB!)

⚠️ Note: For very long videos (1+ hour 4K), consider saving
intermediate segments instead of loading all at once.
    """

    def concatenate(self, session_id: str, output_dir: str = "", cleanup_chunks: bool = True):
        """
        Load and concatenate all chunks into final video.

        Args:
            session_id: Session identifier from ChunkVideoSaver
            output_dir: Output directory (use default if empty)
            cleanup_chunks: Delete chunk files after concatenation

        Returns:
            video: (N, H, W, C) full video tensor
            total_frames: Total number of frames
            stats: JSON string with processing statistics

        Raises:
            FileNotFoundError: If session or manifest not found
            ValueError: If chunks have inconsistent shapes
        """
        if not session_id:
            raise ValueError("session_id is required. Provide the session_id from ChunkVideoSaver.")

        # Determine session directory
        if not output_dir:
            output_dir = folder_paths.get_output_directory()

        session_dir = os.path.join(output_dir, "tlbvfi_chunks", session_id)

        if not os.path.exists(session_dir):
            raise FileNotFoundError(
                f"Session directory not found: {session_dir}\n"
                f"Make sure session_id '{session_id}' is correct and chunks were saved."
            )

        # Load manifest
        manifest_path = os.path.join(session_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                f"Session may be corrupted or incomplete."
            )

        manifest = load_manifest(session_dir)
        chunks_info = sorted(manifest['chunks'], key=lambda c: c['chunk_id'])

        if not chunks_info:
            raise ValueError(f"No chunks found in session {session_id}")

        print(f"\nVideoConcatenator: Loading {len(chunks_info)} chunks from {session_id}")

        # Streaming concatenation to minimize peak memory
        all_frames = []
        total_frames = 0
        expected_shape = None

        # Progress bar
        pbar = tqdm(chunks_info, desc="Loading chunks", unit="chunk")

        for chunk_info in pbar:
            chunk_id = chunk_info['chunk_id']
            chunk_path = chunk_info['path']

            # Validate chunk exists
            if not os.path.exists(chunk_path):
                raise FileNotFoundError(
                    f"Chunk file not found: {chunk_path}\n"
                    f"Chunk {chunk_id} may have been deleted or moved."
                )

            # Load chunk
            try:
                chunk_data = torch.load(chunk_path, map_location='cpu')
                frames = chunk_data['frames']
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load chunk {chunk_id} from {chunk_path}: {e}\n"
                    f"Chunk may be corrupted."
                )

            # Validate shape consistency
            if expected_shape is None:
                expected_shape = frames.shape[1:]  # (H, W, C)
            else:
                if frames.shape[1:] != expected_shape:
                    raise ValueError(
                        f"Chunk {chunk_id} has inconsistent shape:\n"
                        f"  Expected: {expected_shape}\n"
                        f"  Got: {frames.shape[1:]}\n"
                        f"All chunks must have same resolution and channels."
                    )

            # Handle overlapping frames
            # Each chunk includes last frame from previous chunk as first frame
            if len(all_frames) > 0 and chunk_id > 0:
                # Skip first frame to avoid duplication
                frames = frames[1:]

            all_frames.append(frames)
            total_frames += frames.shape[0]

            pbar.set_postfix({'frames': total_frames})

        pbar.close()

        # Concatenate all chunks
        print(f"VideoConcatenator: Concatenating {len(all_frames)} chunks...")
        video = torch.cat(all_frames, dim=0)

        # Generate stats
        session_stats = get_session_stats(session_dir)
        stats = {
            'session_id': session_id,
            'total_chunks': len(chunks_info),
            'total_frames': total_frames,
            'resolution': f"{video.shape[2]}x{video.shape[1]}",  # W×H
            'dtype': str(video.dtype),
            'created_at': session_stats['created_at'],
        }
        stats_json = json.dumps(stats, indent=2)

        # Cleanup
        if cleanup_chunks:
            print(f"VideoConcatenator: Cleaning up chunks in {session_dir}")
            cleanup_session(session_dir, delete_chunks=True, delete_manifest=True)

        print(f"VideoConcatenator: Complete! Final video: {total_frames} frames @ {video.shape[2]}×{video.shape[1]}\n")

        return (video, total_frames, stats_json)
