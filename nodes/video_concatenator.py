"""
VideoConcatenator - Video Chunk Merging

Concatenates video-encoded chunks using FFmpeg concat demuxer (no re-encoding).
Works with ChunkVideoSaver for efficient disk-based video processing.
"""

import torch
import os
import json
import sys
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm

# Use parent package relative import to avoid conflicts with ComfyUI's utils
try:
    from ..utils import (
        load_manifest,
        get_session_stats,
        cleanup_session,
    )
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import (
        load_manifest,
        get_session_stats,
        cleanup_session,
    )

import folder_paths


def find_ffmpeg():
    """Find FFmpeg executable."""
    ffmpeg_paths = [
        'ffmpeg',
        '/usr/bin/ffmpeg',
        '/usr/local/ffmpeg',
        shutil.which('ffmpeg'),
    ]

    for path in ffmpeg_paths:
        if path and (shutil.which(path if path == 'ffmpeg' else None) or (path != 'ffmpeg' and os.path.exists(path))):
            return path if path == 'ffmpeg' else path

    return 'ffmpeg'


class VideoConcatenator:
    """
    Concatenate video-encoded chunks using FFmpeg concat demuxer.

    No re-encoding means:
    - Fast (only demuxing, no decode/encode)
    - Lossless (no quality degradation)
    - Efficient (minimal CPU/GPU usage)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_id": ("STRING", {"default": ""}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": ""}),
                "output_filename": ("STRING", {"default": ""}),  # Auto-generate if empty
                "cleanup_chunks": ("BOOLEAN", {"default": True}),
                "return_frames": ("BOOLEAN", {"default": False}),  # Load frames into memory
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "IMAGE")
    RETURN_NAMES = ("video_path", "total_frames", "stats", "frames")
    FUNCTION = "concatenate"
    CATEGORY = "frame_interpolation/TLBVFI-TF32/chunk"
    OUTPUT_NODE = True

    DESCRIPTION = """
Concatenate video chunks using FFmpeg (no re-encoding).

📌 Purpose:
- Merges all video chunks into final video file
- Uses FFmpeg concat demuxer (fast, lossless)
- No re-encoding = no quality loss
- Final step in chunk-based workflow

🎯 Usage:
1. Enter session_id from ChunkVideoSaver
2. Set output_filename (optional, auto-generated if empty)
3. Set cleanup_chunks=True to delete chunks after merge
4. Set return_frames=False for video file only (recommended)
5. Set return_frames=True to also load frames into memory

⚡ Features:
- FFmpeg concat demuxer: No re-encoding required
- Fast: Only demuxing overhead (~seconds for hours of video)
- Lossless: Bit-perfect copy of encoded chunks
- Automatic chunk validation and ordering

💾 Output:
- Video file: Same codec as chunks (H.264/H.265)
- Size: Sum of chunk sizes + minimal container overhead
- Format: MP4 (playable everywhere)

⚠️ Memory:
- return_frames=False: Minimal memory (~MB)
- return_frames=True: Full video in RAM (use for short videos only)

📊 Performance:
- 1800 chunks: ~10-30 seconds to concat
- No GPU needed (pure demuxing operation)
    """

    def concatenate(self, session_id: str, output_dir: str = "",
                   output_filename: str = "", cleanup_chunks: bool = True,
                   return_frames: bool = False):
        """
        Concatenate video chunks using FFmpeg concat demuxer.

        Args:
            session_id: Session identifier from ChunkVideoSaver
            output_dir: Output directory (use default if empty)
            output_filename: Output filename (auto-generate if empty)
            cleanup_chunks: Delete chunk files after concatenation
            return_frames: Load final video into memory as IMAGE tensor

        Returns:
            video_path: Path to final concatenated video
            total_frames: Total number of frames
            stats: JSON string with statistics
            frames: IMAGE tensor if return_frames=True, else dummy tensor
        """
        if not session_id:
            raise ValueError("session_id is required. Provide the session_id from ChunkVideoSaver.")

        # Determine directories
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

        print(f"\nVideoConcatenator: Concatenating {len(chunks_info)} video chunks from {session_id}")

        # Validate all chunks exist
        for chunk_info in chunks_info:
            chunk_path = chunk_info['path']
            if not os.path.exists(chunk_path):
                raise FileNotFoundError(
                    f"Chunk file not found: {chunk_path}\n"
                    f"Chunk {chunk_info['chunk_id']} may have been deleted or moved."
                )

        # Generate output filename
        if not output_filename:
            output_filename = f"{session_id}_final.mp4"

        output_path = os.path.join(output_dir, output_filename)

        # Create concat list file for FFmpeg
        concat_list_path = os.path.join(session_dir, "concat_list.txt")
        with open(concat_list_path, 'w') as f:
            for chunk_info in chunks_info:
                # FFmpeg concat demuxer format
                f.write(f"file '{os.path.abspath(chunk_info['path'])}'\n")

        # Find FFmpeg
        ffmpeg_path = find_ffmpeg()

        # FFmpeg concat command (no re-encoding)
        cmd = [
            ffmpeg_path,
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_list_path,
            '-c', 'copy',  # IMPORTANT: No re-encoding!
            '-y',  # Overwrite output
            output_path
        ]

        print(f"VideoConcatenator: Running FFmpeg concat demuxer...")

        try:
            # Run FFmpeg with progress
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            stdout, stderr = process.communicate()

            if process.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg concat failed:\n{stderr}"
                )

        except FileNotFoundError:
            raise RuntimeError(
                f"FFmpeg not found. Please install FFmpeg:\n"
                f"  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
                f"  macOS: brew install ffmpeg\n"
                f"  Windows: Download from https://ffmpeg.org/"
            )

        # Calculate total frames
        total_frames = sum(chunk['num_frames'] for chunk in chunks_info)

        # Get file size
        output_size_mb = os.path.getsize(output_path) / (1024**2)

        # Generate stats
        session_stats = get_session_stats(session_dir)
        stats = {
            'session_id': session_id,
            'total_chunks': len(chunks_info),
            'total_frames': total_frames,
            'output_path': output_path,
            'output_size_mb': round(output_size_mb, 1),
            'created_at': session_stats['created_at'],
        }
        stats_json = json.dumps(stats, indent=2)

        print(f"VideoConcatenator: Concatenation complete!")
        print(f"  Output: {output_path}")
        print(f"  Frames: {total_frames}")
        print(f"  Size: {output_size_mb:.1f}MB")

        # Optional: Load frames into memory
        frames = None
        if return_frames:
            print(f"VideoConcatenator: Loading video into memory...")
            frames = self._load_video_to_tensor(output_path)
            print(f"VideoConcatenator: Loaded {frames.shape[0]} frames into memory")
        else:
            # Return dummy tensor
            frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        # Cleanup
        os.remove(concat_list_path)

        if cleanup_chunks:
            print(f"VideoConcatenator: Cleaning up chunks in {session_dir}")
            cleanup_session(session_dir, delete_chunks=True, delete_manifest=True)

        # Return outputs with UI display
        return {
            "ui": {
                "text": [f"Video saved: {output_path}\nFrames: {total_frames}\nSize: {output_size_mb:.1f}MB"]
            },
            "result": (output_path, total_frames, stats_json, frames)
        }

    def _load_video_to_tensor(self, video_path: str) -> torch.Tensor:
        """
        Load video file into IMAGE tensor using FFmpeg.

        Args:
            video_path: Path to video file

        Returns:
            IMAGE tensor (N, H, W, C) in range [0, 1]
        """
        # Get video info
        ffmpeg_path = find_ffmpeg()
        ffprobe_path = ffmpeg_path.replace('ffmpeg', 'ffprobe')

        # Get video properties
        cmd = [
            ffprobe_path,
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,nb_frames',
            '-of', 'json',
            video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        stream = info['streams'][0]

        width = int(stream['width'])
        height = int(stream['height'])
        # nb_frames might not be available, estimate from duration
        num_frames = int(stream.get('nb_frames', 0))

        # Read video frames
        cmd = [
            ffmpeg_path,
            '-i', video_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-'
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw_data, _ = process.communicate()

        # Convert to numpy array
        import numpy as np
        frames = np.frombuffer(raw_data, dtype=np.uint8)

        # Reshape
        num_frames_actual = len(frames) // (width * height * 3)
        frames = frames.reshape((num_frames_actual, height, width, 3))

        # Convert to torch tensor [0, 1]
        frames_tensor = torch.from_numpy(frames).float() / 255.0

        return frames_tensor
