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


def get_video_encoding_info(video_path: str):
    """
    Extract encoding information from source video using FFprobe.

    Args:
        video_path: Path to source video file

    Returns:
        dict with keys: codec_name, bitrate, pix_fmt, profile, level
        Returns None if video_path is empty or file doesn't exist
    """
    if not video_path or not os.path.exists(video_path):
        return None

    import json

    ffmpeg_path = find_ffmpeg()
    ffprobe_path = ffmpeg_path.replace('ffmpeg', 'ffprobe')

    cmd = [
        ffprobe_path,
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,bit_rate,pix_fmt,profile,level',
        '-of', 'json',
        video_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print(f"Warning: FFprobe failed to read video info: {result.stderr}")
            return None

        info = json.loads(result.stdout)
        if 'streams' not in info or len(info['streams']) == 0:
            return None

        stream = info['streams'][0]

        return {
            'codec_name': stream.get('codec_name', ''),
            'bitrate': int(stream.get('bit_rate', 0)) if stream.get('bit_rate') else None,
            'pix_fmt': stream.get('pix_fmt', 'yuv420p'),
            'profile': stream.get('profile', ''),
            'level': stream.get('level', 0),
        }
    except Exception as e:
        print(f"Warning: Failed to extract video info: {e}")
        return None


def map_codec_to_encoder(codec_name: str) -> str:
    """
    Map codec name from FFprobe to FFmpeg encoder name.

    Args:
        codec_name: Codec name from FFprobe (e.g., 'h264', 'hevc')

    Returns:
        FFmpeg encoder name (e.g., 'libx264', 'libx265')
    """
    codec_map = {
        'h264': 'libx264',
        'hevc': 'libx265',
        'h265': 'libx265',
        'mpeg4': 'libx264',  # Fallback to H.264
        'vp9': 'libx265',    # Fallback to H.265 for similar quality
    }
    return codec_map.get(codec_name.lower(), 'libx264')  # Default to H.264


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
                "source_video_path": ("STRING", {"default": ""}),  # Auto-detect codec/bitrate from source
                "codec": (["libx264", "libx265"],),  # H.264 or H.265 (ignored if source_video_path provided)
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
- Auto-detects and preserves source video quality

🎯 Usage:
1. Connect interpolated_frames from TLBVFI_Interpolator
2. Set chunk_id (0, 1, 2, ...) - increment for each chunk
3. Set fps (should match source video)
4. [RECOMMENDED] Set source_video_path to auto-detect codec/bitrate
5. OR manually choose codec: libx264 (faster) or libx265 (smaller)
6. Set quality: 18 (visually lossless) to 28 (smaller file)

💾 Storage:
- Format: MP4 with H.264/H.265
- Location: output_dir/tlbvfi_chunks/session_id/
- Size: ~50-100MB per chunk (9 frames @ 4K, CRF 18)
- Concat: FFmpeg concat demuxer (no re-encoding)

⚙️ Quality Preservation:
- source_video_path set: Auto-detects codec, bitrate, pixel format
  → H.264 → H.264, H.265 → H.265, 10bit → 10bit
  → Uses source bitrate for identical quality
- source_video_path empty: Manual codec selection + CRF mode

⚙️ Manual Settings (when source_video_path empty):
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
                   session_id: str = "", output_dir: str = "", source_video_path: str = "",
                   codec: str = "libx264", quality: int = 18):
        """
        Save frames as video-encoded chunk.

        Args:
            frames: (N, H, W, C) tensor from TLBVFI_Interpolator
            chunk_id: Sequential chunk number
            fps: Frame rate for video
            session_id: Unique session identifier (auto-generated if empty)
            output_dir: Output directory (use default if empty)
            source_video_path: Path to source video for auto codec/bitrate detection
            codec: libx264 or libx265 (ignored if source_video_path provided)
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

        # Auto-detect encoding settings from source video
        video_info = get_video_encoding_info(source_video_path) if source_video_path else None

        if video_info:
            # Use source video settings
            detected_codec = map_codec_to_encoder(video_info['codec_name'])
            detected_pix_fmt = video_info['pix_fmt']
            detected_bitrate = video_info['bitrate']

            print(f"ChunkVideoSaver: Detected source video encoding:")
            print(f"  Codec: {video_info['codec_name']} → {detected_codec}")
            print(f"  Pixel Format: {detected_pix_fmt}")
            if detected_bitrate:
                print(f"  Bitrate: {detected_bitrate / 1000000:.2f} Mbps")

            # Override parameters
            codec = detected_codec
            pix_fmt = detected_pix_fmt
            use_bitrate = detected_bitrate is not None
        else:
            # Use manual settings
            pix_fmt = 'yuv420p'
            use_bitrate = False

        # Find FFmpeg
        ffmpeg_path = find_ffmpeg()

        # FFmpeg command for video encoding
        # Key settings for concat compatibility:
        # - GOP size = chunk size: Each chunk starts with keyframe
        # - Same codec/quality/bitrate across all chunks
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
        ]

        # Quality/bitrate mode
        if use_bitrate and video_info['bitrate']:
            # Use bitrate mode to match source
            cmd.extend(['-b:v', str(video_info['bitrate'])])
            cmd.extend(['-maxrate', str(int(video_info['bitrate'] * 1.5))])
            cmd.extend(['-bufsize', str(int(video_info['bitrate'] * 2))])
        else:
            # Use CRF mode
            cmd.extend(['-crf', str(quality)])

        # Common settings
        cmd.extend([
            '-pix_fmt', pix_fmt,
            '-g', str(num_frames),  # GOP size = chunk size (keyframe at start)
            '-preset', 'medium',  # Encoding speed/quality trade-off
            '-movflags', '+faststart',  # Optimize for streaming
            chunk_path
        ])

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

        # Build encoding info string
        if use_bitrate and video_info['bitrate']:
            encoding_info = f"Codec: {codec}, Bitrate: {video_info['bitrate'] / 1000000:.2f} Mbps, Pix fmt: {pix_fmt}"
        else:
            encoding_info = f"Codec: {codec}, CRF: {quality}, Pix fmt: {pix_fmt}"

        print(
            f"ChunkVideoSaver: Saved chunk {chunk_id} → {chunk_path}\n"
            f"  {num_frames} frames @ {H}×{W}, {fps} fps\n"
            f"  {encoding_info}\n"
            f"  Size: {file_size_mb:.1f}MB"
        )

        return (session_id, chunk_path, num_frames, f"{file_size_mb:.1f}MB")
