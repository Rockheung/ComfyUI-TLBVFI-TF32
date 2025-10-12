"""
ChunkVideoSaver - Video-Encoded Chunks

Saves interpolated frames as H.264/H.265 encoded video chunks.
Efficient disk usage with concat-compatible MP4 files.
"""

import torch
import os
import sys
import subprocess
import shutil
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    create_session_id,
    add_chunk_to_manifest,
)

import folder_paths


def find_ffmpeg():
    """Find FFmpeg executable."""
    # Try common locations
    ffmpeg_paths = [
        'ffmpeg',  # System PATH
        '/usr/bin/ffmpeg',
        '/usr/local/bin/ffmpeg',
        shutil.which('ffmpeg'),
    ]

    for path in ffmpeg_paths:
        if path and shutil.which(path if path == 'ffmpeg' else None) or (path != 'ffmpeg' and os.path.exists(path)):
            return path if path == 'ffmpeg' else path

    # If nothing found, return 'ffmpeg' and hope it's in PATH
    return 'ffmpeg'


class ChunkVideoSaver:
    """
    Save interpolated frames as H.264/H.265 encoded video chunks.

    Disk usage per chunk (9 frames @ 4K):
    - H.264 CRF18: ~50-100MB
    - H.265 CRF23: ~30-50MB

    Chunks are concat-compatible (no re-encoding needed).
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
                "fps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                }),
            },
            "optional": {
                "session_id": ("STRING", {"default": ""}),  # Auto-generate if empty
                "output_dir": ("STRING", {"default": ""}),  # Use ComfyUI default if empty
                "codec": (["libx264", "libx265"],),  # H.264 or H.265
                "quality": ("INT", {
                    "default": 18,  # CRF value (lower = better quality)
                    "min": 0,
                    "max": 51,
                    "step": 1,
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("session_id", "chunk_path", "num_frames", "file_size_mb")
    FUNCTION = "save_chunk"
    CATEGORY = "frame_interpolation/TLBVFI-TF32/chunk"

    DESCRIPTION = """
Save interpolated frames as video-encoded chunks (H.264/H.265).

📌 Purpose:
- Encodes frames to H.264/H.265 for efficient disk usage
- Concat-compatible (no re-encoding needed for final video)
- Each chunk is independently playable

🎯 Usage:
1. Connect interpolated_frames from TLBVFI_Interpolator
2. Set chunk_id (0, 1, 2, ...) - increment for each chunk
3. Set fps (should match source video)
4. Choose codec: libx264 (faster) or libx265 (smaller)
5. Set quality: 18 (visually lossless) to 28 (smaller file)

💾 Storage:
- Format: MP4 with H.264/H.265
- Location: output_dir/tlbvfi_chunks/session_id/
- Size: ~50-100MB per chunk (9 frames @ 4K, CRF 18)
- Concat: FFmpeg concat demuxer (no re-encoding)

⚙️ Settings:
- codec=libx264: Faster encoding, good compatibility
- codec=libx265: Better compression, smaller files
- quality=18: Visually lossless (recommended)
- quality=23: Good balance
- quality=28: Smaller files, slight quality loss

📊 Disk usage (1800 chunks @ 4K):
- H.264 CRF18: 90-180 GB
- H.265 CRF23: 54-90 GB
    """

    def save_chunk(self, frames: torch.Tensor, chunk_id: int, fps: int = 30,
                   session_id: str = "", output_dir: str = "",
                   codec: str = "libx264", quality: int = 18):
        """
        Save frames as video-encoded chunk.

        Args:
            frames: (N, H, W, C) tensor from TLBVFI_Interpolator
            chunk_id: Sequential chunk number
            fps: Frame rate for video
            session_id: Unique session identifier (auto-generated if empty)
            output_dir: Output directory (use default if empty)
            codec: libx264 or libx265
            quality: CRF value (0-51, lower = better)

        Returns:
            session_id: Echo back for workflow tracking
            chunk_path: Absolute path to saved chunk
            num_frames: Number of frames in chunk
            file_size_mb: File size in MB
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

        # Save chunk as video file
        chunk_filename = f"chunk_{chunk_id:06d}.mp4"
        chunk_path = os.path.join(session_dir, chunk_filename)

        # Convert tensor to numpy (uint8)
        frames_cpu = frames.cpu() if frames.device.type != 'cpu' else frames
        frames_np = (frames_cpu.numpy() * 255).clip(0, 255).astype(np.uint8)

        num_frames, H, W, C = frames_np.shape

        # Find FFmpeg
        ffmpeg_path = find_ffmpeg()

        # FFmpeg command for video encoding
        # Key settings for concat compatibility:
        # - yuv420p: Standard pixel format
        # - GOP size = chunk size: Each chunk starts with keyframe
        # - Same codec/quality across all chunks
        cmd = [
            ffmpeg_path,
            '-y',  # Overwrite output file
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{W}x{H}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-',  # Read from stdin
            '-an',  # No audio
            '-vcodec', codec,
            '-crf', str(quality),
            '-pix_fmt', 'yuv420p',  # Standard format for compatibility
            '-g', str(num_frames),  # GOP size = chunk size (keyframe at start)
            '-preset', 'medium',  # Encoding speed/quality trade-off
            '-movflags', '+faststart',  # Optimize for streaming
            chunk_path
        ]

        # Run FFmpeg
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Write frames to FFmpeg stdin
            stdout, stderr = process.communicate(input=frames_np.tobytes())

            if process.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg encoding failed:\n{stderr.decode('utf-8', errors='ignore')}"
                )

        except FileNotFoundError:
            raise RuntimeError(
                f"FFmpeg not found. Please install FFmpeg:\n"
                f"  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
                f"  macOS: brew install ffmpeg\n"
                f"  Windows: Download from https://ffmpeg.org/"
            )

        # Get file size
        file_size_bytes = os.path.getsize(chunk_path)
        file_size_mb = file_size_bytes / (1024**2)

        # Update manifest
        add_chunk_to_manifest(
            session_dir=session_dir,
            chunk_id=chunk_id,
            chunk_path=chunk_path,
            shape=frames_np.shape,
            status='complete'
        )

        print(
            f"ChunkVideoSaver: Saved chunk {chunk_id} → {chunk_path}\n"
            f"  {num_frames} frames @ {H}×{W}, {fps} fps\n"
            f"  Codec: {codec}, CRF: {quality}\n"
            f"  Size: {file_size_mb:.1f}MB"
        )

        return (session_id, chunk_path, num_frames, f"{file_size_mb:.1f}MB")
