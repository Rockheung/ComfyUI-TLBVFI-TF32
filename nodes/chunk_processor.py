"""
TLBVFI_ChunkProcessor - All-in-One Chunk-Based Video Interpolation

Processes entire video by automatically iterating through all frame pairs,
interpolating, and saving video-encoded chunks progressively.

This node replaces the manual workflow:
  FramePairSlicer → TLBVFI_Interpolator → ChunkVideoSaver (repeated N times)

With a single automated node that handles the entire pipeline internally.
"""

import torch
import gc
import sys
import os
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Use parent package relative import
try:
    from ..utils import (
        load_tlbvfi_model,
        enable_tf32_if_available,
        enable_cudnn_benchmark,
        cleanup_memory,
        get_memory_stats,
        print_memory_summary,
        create_session_id,
        create_manifest,
        save_manifest,
        load_manifest,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import (
        load_tlbvfi_model,
        enable_tf32_if_available,
        enable_cudnn_benchmark,
        cleanup_memory,
        get_memory_stats,
        print_memory_summary,
        create_session_id,
        create_manifest,
        save_manifest,
        load_manifest,
    )

import folder_paths


def find_models(folder_type: str, extensions: list) -> list:
    """Find all model files with given extensions in the specified folder type."""
    model_list = []
    base_paths = folder_paths.get_folder_paths(folder_type)

    for base_path in base_paths:
        for root, _, files in os.walk(base_path, followlinks=True):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    relative_path = os.path.relpath(os.path.join(root, file), base_path)
                    model_list.append(relative_path.replace("\\", "/"))
    return sorted(list(set(model_list)))


def find_ffmpeg():
    """Find FFmpeg executable."""
    ffmpeg_paths = [
        'ffmpeg',
        '/usr/bin/ffmpeg',
        '/usr/local/bin/ffmpeg',
        shutil.which('ffmpeg'),
    ]

    for path in ffmpeg_paths:
        if path and (shutil.which(path if path == 'ffmpeg' else None) or (path != 'ffmpeg' and os.path.exists(path))):
            return path if path == 'ffmpeg' else path

    return 'ffmpeg'


class TLBVFI_ChunkProcessor:
    """
    All-in-one chunk-based video interpolation processor.

    Automatically iterates through all frame pairs, interpolates them,
    and saves video-encoded chunks progressively to disk.
    """

    @classmethod
    def INPUT_TYPES(cls):
        unet_models = find_models("interpolation", [".pth"])
        if not unet_models:
            raise Exception(
                "No TLBVFI UNet models (.pth) found in 'ComfyUI/models/interpolation/'. "
                "Please download 'vimeo_unet.pth'."
            )

        return {
            "required": {
                "images": ("IMAGE",),  # (N, H, W, C) from VHS LoadVideo
                "model_name": (unet_models,),
                "times_to_interpolate": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "codec": (["h264_nvenc", "hevc_nvenc", "libx264", "libx265", "libvpx-vp9"],),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51, "step": 1}),
                "gpu_id": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "session_id": ("STRING", {"default": ""}),  # Auto-generate if empty
                "save_debug_images": ("BOOLEAN", {"default": False}),  # Save PNG frames for debugging
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("session_id",)
    FUNCTION = "process_all_chunks"
    CATEGORY = "frame_interpolation/TLBVFI-TF32/chunk"

    DESCRIPTION = """
TLBVFI all-in-one chunk processor - automatically processes entire video.

📌 Purpose:
- Processes ALL frame pairs automatically (no manual iteration needed)
- Interpolates each pair and saves as video chunk progressively
- Model stays loaded in VRAM for maximum efficiency
- Memory-safe: Only processes 2 frames at a time

🎯 Usage:
1. Connect IMAGE tensor from VHS LoadVideo
2. Select model (vimeo_unet.pth)
3. Set times_to_interpolate (1=2x, 2=4x, 3=8x, 4=16x)
4. Set video encoding parameters (fps, codec, crf)
5. Run once - all pairs will be processed automatically
6. Connect session_id output to VideoConcatenator

⚡ Features:
- Automatic iteration through all frame pairs
- Progressive disk saving (no RAM accumulation)
- Model reuse across all pairs (no reload overhead)
- TF32 acceleration on RTX 30/40 series
- Progress bar with ETA

💾 Output:
- Video chunks: output/tlbvfi_chunks/SESSION_ID/chunk_XXXX.mp4
- Manifest: output/tlbvfi_chunks/SESSION_ID/manifest.json
- session_id: For VideoConcatenator to merge chunks

📊 Performance:
- 1800 frame pairs: ~20 hours (3.6GB model loaded once)
- Each chunk: ~30-60 seconds (interpolation + encoding)
- Memory: ~13GB GPU for 4K video

🐛 Debug:
- save_debug_images=True: Saves PNG frames for each pair
    """

    def process_all_chunks(self, images: torch.Tensor, model_name: str, times_to_interpolate: int,
                          fps: float, codec: str, crf: int, gpu_id: int,
                          session_id: str = "", save_debug_images: bool = False):
        """
        Process all frame pairs automatically.

        Args:
            images: (N, H, W, C) tensor from VHS LoadVideo
            model_name: Model checkpoint name
            times_to_interpolate: 1=2x, 2=4x, 3=8x, 4=16x
            fps: Output video framerate
            codec: Video codec (h264_nvenc, hevc_nvenc, etc.)
            crf: Quality (0=lossless, 51=worst, 18=visually lossless)
            gpu_id: CUDA device ID
            session_id: Session identifier (auto-generated if empty)
            save_debug_images: Save PNG frames for debugging

        Returns:
            session_id: For VideoConcatenator
        """
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        N = images.shape[0]
        H, W = images.shape[1:3]
        total_pairs = N - 1

        if N < 2:
            raise ValueError(
                f"TLBVFI_ChunkProcessor requires at least 2 frames, got {N}. "
                f"Please load a video with multiple frames."
            )

        print(f"\n{'='*80}")
        print(f"TLBVFI_ChunkProcessor: Starting processing")
        print(f"  Input: {N} frames @ {H}×{W}")
        print(f"  Total pairs to process: {total_pairs}")
        print(f"  Interpolation: {times_to_interpolate}x ({2**times_to_interpolate}x output frames)")
        print(f"  Output FPS: {fps}")
        print(f"  Codec: {codec} (CRF={crf})")
        print(f"{'='*80}\n")

        # Create session
        if not session_id:
            session_id = datetime.now().strftime("tlbvfi_%Y%m%d_%H%M%S")

        output_dir = folder_paths.get_output_directory()
        session_dir = os.path.join(output_dir, "tlbvfi_chunks", session_id)
        os.makedirs(session_dir, exist_ok=True)

        print(f"TLBVFI_ChunkProcessor: Session directory: {session_dir}\n")

        # Load model ONCE
        print(f"TLBVFI_ChunkProcessor: Loading model {model_name}")
        print_memory_summary(device, "Before model load: ")

        model = load_tlbvfi_model(model_name, device)
        enable_tf32_if_available(device)
        enable_cudnn_benchmark(device)

        print_memory_summary(device, "After model load: ")
        print()

        # Process all pairs
        ffmpeg_path = find_ffmpeg()
        manifest = {'chunks': [], 'metadata': {
            'session_id': session_id,
            'total_input_frames': N,
            'total_pairs': total_pairs,
            'times_to_interpolate': times_to_interpolate,
            'fps': fps,
            'codec': codec,
            'crf': crf,
            'resolution': f"{H}x{W}",
        }}

        for pair_idx in tqdm(range(total_pairs), desc="Processing frame pairs"):
            is_last_pair = (pair_idx == total_pairs - 1)

            print(f"\n{'-'*80}")
            print(f"TLBVFI_ChunkProcessor: Processing pair {pair_idx+1}/{total_pairs} "
                  f"(frames {pair_idx}-{pair_idx+1})")
            if is_last_pair:
                print(f"  [LAST PAIR - will include end frame]")
            print(f"{'-'*80}")

            # Extract frame pair
            frame_pair = images[pair_idx:pair_idx+2]  # (2, H, W, C)

            # Interpolate
            interpolated_frames = self._interpolate_pair(
                frame_pair, model, times_to_interpolate, device, is_last_pair, save_debug_images, pair_idx
            )

            # Save as video chunk
            chunk_path = os.path.join(session_dir, f"chunk_{pair_idx:04d}.mp4")
            self._save_chunk_as_video(
                interpolated_frames, chunk_path, fps, codec, crf, ffmpeg_path
            )

            # Update manifest
            manifest['chunks'].append({
                'chunk_id': pair_idx,
                'path': chunk_path,
                'num_frames': interpolated_frames.shape[0],
                'is_last_pair': is_last_pair,
            })

            # Save manifest progressively
            save_manifest(manifest, session_dir)

            # Cleanup
            del interpolated_frames, frame_pair
            cleanup_memory(device, force_gc=True)

        print(f"\n{'='*80}")
        print(f"TLBVFI_ChunkProcessor: Complete!")
        print(f"  Processed {total_pairs} pairs")
        print(f"  Output: {session_dir}")
        print(f"  Session ID: {session_id}")
        print(f"{'='*80}\n")

        return (session_id,)

    def _interpolate_pair(self, frame_pair: torch.Tensor, model, times_to_interpolate: int,
                         device, is_last_pair: bool, save_debug_images: bool, pair_idx: int):
        """
        Interpolate a single frame pair.

        Returns:
            interpolated_frames: (N, H, W, C) tensor
        """
        # Preprocessing: (2, H, W, C) -> (2, C, H, W), normalize to [-1, 1]
        image_tensors = frame_pair.permute(0, 3, 1, 2).float()
        image_tensors = (image_tensors * 2.0) - 1.0

        # Transfer to GPU
        frame1 = image_tensors[0].unsqueeze(0).to(device, non_blocking=True)
        frame2 = image_tensors[1].unsqueeze(0).to(device, non_blocking=True)

        # Interpolation loop
        current_frames = [frame1, frame2]
        for iteration in range(times_to_interpolate):
            temp_frames = [current_frames[0]]
            for j in range(len(current_frames) - 1):
                with torch.no_grad():
                    mid_frame = model.sample(current_frames[j], current_frames[j+1], disable_progress=True)
                temp_frames.extend([mid_frame, current_frames[j+1]])
            current_frames = temp_frames

            num_frames = len(current_frames)
            print(f"  Iteration {iteration+1}/{times_to_interpolate}: Generated {num_frames} frames")

        # Post-processing: back to ComfyUI format (N, H, W, C)
        # Exclude last frame unless it's the last pair of the video
        frames_to_process = current_frames if is_last_pair else current_frames[:-1]

        processed_frames = []
        for frame in frames_to_process:
            # Move to CPU, denormalize
            frame_cpu = frame.squeeze(0).to('cpu', non_blocking=True)
            frame_cpu = (frame_cpu + 1.0) / 2.0
            frame_cpu = frame_cpu.clamp(0, 1)
            frame_cpu = frame_cpu.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
            processed_frames.append(frame_cpu)

        result = torch.stack(processed_frames, dim=0)  # (N, H, W, C)

        # Optional: Save frames as PNG images
        if save_debug_images:
            from PIL import Image
            import numpy as np

            output_dir = folder_paths.get_output_directory()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(output_dir, f"tlbvfi_frames_{timestamp}", f"pair_{pair_idx:04d}")
            os.makedirs(save_dir, exist_ok=True)

            for idx, frame_tensor in enumerate(processed_frames):
                # Convert to numpy uint8
                frame_np = (frame_tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(frame_np)
                img_path = os.path.join(save_dir, f"frame_{idx:04d}.png")
                img.save(img_path)

            print(f"  Saved {len(processed_frames)} frames to {save_dir}")

        # Memory cleanup
        del current_frames, temp_frames, frame1, frame2, image_tensors, processed_frames

        print(f"  Generated {result.shape[0]} frames from 2 input frames")
        if not is_last_pair:
            print(f"  (Excluded end frame to avoid duplication in concat)")

        return result

    def _save_chunk_as_video(self, frames: torch.Tensor, chunk_path: str,
                            fps: float, codec: str, crf: int, ffmpeg_path: str):
        """
        Save frames as H.264/H.265 encoded video using FFmpeg.

        Args:
            frames: (N, H, W, C) tensor in range [0, 1]
            chunk_path: Output video path
            fps: Framerate
            codec: Video codec
            crf: Quality (0-51)
            ffmpeg_path: FFmpeg executable path
        """
        import numpy as np

        num_frames, H, W, C = frames.shape

        # Convert to uint8
        frames_np = (frames.numpy() * 255).clip(0, 255).astype(np.uint8)

        # Determine pixel format
        pix_fmt = 'yuv420p' if codec in ['h264_nvenc', 'hevc_nvenc', 'libx264', 'libx265'] else 'yuv420p'

        # GOP size must be >= 2 for NVENC
        gop_size = max(num_frames, 2)

        # FFmpeg command
        cmd = [
            ffmpeg_path,
            '-y',  # Overwrite
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{W}x{H}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-',  # Read from stdin
            '-c:v', codec,
            '-crf', str(crf),
            '-pix_fmt', pix_fmt,
            '-g', str(gop_size),
            '-bf', '0',  # No B-frames for concat compatibility
            '-preset', 'medium',
            '-movflags', '+faststart',
            chunk_path
        ]

        # Run FFmpeg
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Write frames
        process.stdin.write(frames_np.tobytes())
        process.stdin.close()

        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(
                f"FFmpeg encoding failed for {chunk_path}:\\n{stderr.decode()}"
            )

        # Get file size
        chunk_size_mb = os.path.getsize(chunk_path) / (1024**2)
        print(f"  Saved chunk: {chunk_path} ({num_frames} frames, {chunk_size_mb:.1f}MB)")
