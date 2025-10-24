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
        enable_cudnn_benchmark,
        cleanup_memory,
        save_manifest,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import (
        enable_cudnn_benchmark,
        cleanup_memory,
        save_manifest,
    )

import folder_paths
from .tlbvfi_interpolator_v2 import TLBVFI_Interpolator_V2

try:
    from comfy.utils import should_stop_processing as comfy_should_stop_processing
except ImportError:
    comfy_should_stop_processing = None

try:
    import execution  # type: ignore
except ImportError:
    execution = None


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

    @staticmethod
    def _stop_requested() -> bool:
        """
        Detect whether ComfyUI requested the current prompt to stop.

        Supports both comfy.utils.should_stop_processing (latest builds)
        and legacy execution module fallbacks. Returns False if no hook
        is available or if running outside ComfyUI.
        """
        if comfy_should_stop_processing:
            try:
                if comfy_should_stop_processing():
                    return True
            except Exception:
                # Ignore hook errors and try fallbacks
                pass

        if execution is not None:
            for attr_name in ("should_stop_processing", "should_stop"):
                checker = getattr(execution, attr_name, None)
                if callable(checker):
                    try:
                        if checker():
                            return True
                    except Exception:
                        pass
            stop_flag = getattr(execution, "stop_processing", None)
            if hasattr(stop_flag, "is_set"):
                try:
                    if stop_flag.is_set():
                        return True
                except Exception:
                    pass

        return False

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
                "bitrate": ("STRING", {"default": "50M"}),  # e.g., "50M", "100M"
                "gpu_id": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "session_id": ("STRING", {"default": ""}),  # Auto-generate if empty
                "save_debug_images": ("BOOLEAN", {"default": False}),  # Save PNG frames for debugging
                "enable_tf32": ("BOOLEAN", {"default": True}),
                "sample_steps": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "slider"
                }),
                "flow_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "tile_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2048,
                    "step": 128,
                    "display": "number"
                }),
                # 0 = Auto-calculate optimal tile size based on 80% of available GPU memory
                #     (will disable tiling if image fits in memory or is small enough)
                # 512/640/.../2048 = Manual tile size (must be multiple of 128)
                # Recommended: 0 for automatic optimization
                "cpu_offload": ("BOOLEAN", {"default": True}),
                "debug": ("BOOLEAN", {"default": False}),
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
4. Set video encoding parameters (fps, codec, bitrate)
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
                          fps: float, codec: str, bitrate: str, gpu_id: int,
                          session_id: str = "", save_debug_images: bool = False,
                          enable_tf32: bool = True, sample_steps: int = 10,
                          flow_scale: float = 0.5, tile_size: int = 512, cpu_offload: bool = True,
                          debug: bool = False):
        """
        Process all frame pairs automatically.

        Args:
            images: (N, H, W, C) tensor from VHS LoadVideo
            model_name: Model checkpoint name
            times_to_interpolate: 1=2x, 2=4x, 3=8x, 4=16x
            fps: Output video framerate
            codec: Video codec (h264_nvenc, hevc_nvenc, etc.)
            bitrate: Target bitrate (e.g., "50M", "100M")
            gpu_id: CUDA device ID
            session_id: Session identifier (auto-generated if empty)
            save_debug_images: Save PNG frames for debugging
            enable_tf32: Toggle TF32 acceleration (RTX 30/40)
            sample_steps: Diffusion steps passed to Interpolator V2
            flow_scale: Optical flow resolution scaling
            tile_size: Tile size for tiled inference (0=disabled, 512=recommended for 2K/4K)
            cpu_offload: Offload intermediate frames to CPU after each step

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

        # Calculate input tensor memory
        input_size_gb = images.element_size() * images.nelement() / (1024**3)

        print(f"\n{'='*80}")
        print(f"TLBVFI_ChunkProcessor: Starting processing")
        print(f"  Input: {N} frames @ {H}×{W} ({input_size_gb:.2f}GB in memory)")
        print(f"  Total pairs to process: {total_pairs}")
        print(f"  Interpolation: {times_to_interpolate}x ({2**times_to_interpolate}x output frames)")
        print(f"  Output FPS: {fps}")
        print(f"  Codec: {codec} (Bitrate={bitrate})")
        print(f"  Sample steps: {sample_steps}")
        print(f"  Flow scale: {flow_scale}")
        print(f"  Tile size: {tile_size if tile_size > 0 else 'Auto (will be calculated per frame)'}")
        print(f"  TF32: {'Enabled' if enable_tf32 else 'Disabled'}")
        print(f"  CPU offload: {'Enabled' if cpu_offload else 'Disabled'}")
        print(f"{'='*80}\n")

        # Create session
        if not session_id:
            session_id = datetime.now().strftime("tlbvfi_%Y%m%d_%H%M%S")

        output_dir = folder_paths.get_output_directory()
        session_dir = os.path.join(output_dir, "tlbvfi_chunks", session_id)
        os.makedirs(session_dir, exist_ok=True)

        print(f"TLBVFI_ChunkProcessor: Session directory: {session_dir}\n")

        # Load model via production-grade interpolator (cached)
        print(f"TLBVFI_ChunkProcessor: Loading model {model_name} (sample_steps={sample_steps})")
        interpolator = TLBVFI_Interpolator_V2()
        cache_key = f"{model_name}_{gpu_id}_{sample_steps}"
        interpolator._get_or_load_model(cache_key, model_name, device, sample_steps)
        enable_cudnn_benchmark(device)

        print()

        # Process all pairs
        ffmpeg_path = find_ffmpeg()
        manifest = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'chunks': [],
            'metadata': {
                'total_input_frames': N,
                'total_pairs': total_pairs,
                'times_to_interpolate': times_to_interpolate,
                'fps': fps,
                'codec': codec,
                'bitrate': bitrate,
                'resolution': f"{H}x{W}",
                'sample_steps': sample_steps,
                'flow_scale': flow_scale,
                'enable_tf32': enable_tf32,
                'cpu_offload': cpu_offload,
            }
        }

        interrupted = False
        progress_bar = tqdm(range(total_pairs), desc="Processing frame pairs")
        for pair_idx in progress_bar:
            if self._stop_requested():
                interrupted = True
                print("\nTLBVFI_ChunkProcessor: Stop requested - aborting before processing remaining pairs.")
                break

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
                frame_pair,
                interpolator,
                model_name,
                times_to_interpolate,
                enable_tf32,
                sample_steps,
                flow_scale,
                tile_size,
                cpu_offload,
                gpu_id,
                is_last_pair,
                save_debug_images,
                pair_idx,
                debug
            )

            # Save as video chunk
            chunk_path = os.path.join(session_dir, f"chunk_{pair_idx:04d}.mp4")
            self._save_chunk_as_video(
                interpolated_frames, chunk_path, fps, codec, ffmpeg_path, bitrate
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

            # Log memory usage
            if torch.cuda.is_available():
                gpu_mem_allocated = torch.cuda.memory_allocated(device) / (1024**3)
                gpu_mem_reserved = torch.cuda.memory_reserved(device) / (1024**3)
                print(f"  GPU Memory: {gpu_mem_allocated:.2f}GB allocated, {gpu_mem_reserved:.2f}GB reserved")

            if self._stop_requested():
                interrupted = True
                print("\nTLBVFI_ChunkProcessor: Stop requested - ending after current chunk.")
                break

        progress_bar.close()

        manifest['metadata']['completed_pairs'] = len(manifest['chunks'])
        manifest['metadata']['status'] = 'interrupted' if interrupted else 'completed'
        save_manifest(manifest, session_dir)

        processed_pairs = manifest['metadata']['completed_pairs']
        print(f"\n{'='*80}")
        if interrupted:
            print(f"TLBVFI_ChunkProcessor: Stopped early at pair {processed_pairs}/{total_pairs}.")
            print(f"  Partial chunks saved to: {session_dir}")
            print(f"  Session ID (resume/concat): {session_id}")
        else:
            print(f"TLBVFI_ChunkProcessor: Complete!")
            print(f"  Processed {processed_pairs} pairs")
            print(f"  Output: {session_dir}")
            print(f"  Session ID: {session_id}")
        print(f"{'='*80}\n")

        return (session_id,)

    def _interpolate_pair(self, frame_pair: torch.Tensor,
                         interpolator: TLBVFI_Interpolator_V2,
                         model_name: str,
                         times_to_interpolate: int,
                         enable_tf32: bool,
                         sample_steps: int,
                         flow_scale: float,
                         tile_size: int,
                         cpu_offload: bool,
                         gpu_id: int,
                         is_last_pair: bool,
                         save_debug_images: bool,
                         pair_idx: int,
                         debug: bool = False) -> torch.Tensor:
        """
        Interpolate a single frame pair using the production-grade V2 logic.

        Returns:
            interpolated_frames: (N, H, W, C) tensor on CPU
        """
        prev_frame = frame_pair[0:1]
        next_frame = frame_pair[1:2]

        interpolated_tuple = interpolator.interpolate(
            prev_frame,
            next_frame,
            model_name,
            times_to_interpolate=times_to_interpolate,
            enable_tf32=enable_tf32,
            sample_steps=sample_steps,
            flow_scale=flow_scale,
            tile_size=tile_size,
            cpu_offload=cpu_offload,
            gpu_id=gpu_id,
            debug=debug,
        )

        frames = interpolated_tuple[0].to('cpu', non_blocking=True)

        # Clean up intermediate tensors
        del prev_frame, next_frame, interpolated_tuple
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if not is_last_pair:
            frames = frames[:-1]

        if save_debug_images:
            from PIL import Image
            import numpy as np

            output_dir = folder_paths.get_output_directory()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(output_dir, f"tlbvfi_frames_{timestamp}", f"pair_{pair_idx:04d}")
            os.makedirs(save_dir, exist_ok=True)

            for idx, frame_tensor in enumerate(frames):
                frame_np = (frame_tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(frame_np)
                img_path = os.path.join(save_dir, f"frame_{idx:04d}.png")
                img.save(img_path)
                del frame_np, img

            print(f"  Saved {frames.shape[0]} frames to {save_dir}")

        frames = frames.contiguous()

        print(f"  Generated {frames.shape[0]} frames from 2 input frames")
        if not is_last_pair:
            print(f"  (Excluded end frame to avoid duplication in concat)")

        return frames

    def _save_chunk_as_video(self, frames: torch.Tensor, chunk_path: str,
                            fps: float, codec: str, ffmpeg_path: str, bitrate: str):
        """
        Save frames as H.264/H.265 encoded video using FFmpeg.

        Args:
            frames: (N, H, W, C) tensor in range [0, 1]
            chunk_path: Output video path
            fps: Framerate
            codec: Video codec
            ffmpeg_path: FFmpeg executable path
            bitrate: Target bitrate (e.g., "50M", "100M")
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
            '-b:v', bitrate,
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

        # Explicitly delete large numpy array
        del frames_np
        gc.collect()

        if process.returncode != 0:
            raise RuntimeError(
                f"FFmpeg encoding failed for {chunk_path}:\\n{stderr.decode()}"
            )

        # Get file size
        chunk_size_mb = os.path.getsize(chunk_path) / (1024**2)
        print(f"  Saved chunk: {chunk_path} ({num_frames} frames, {chunk_size_mb:.1f}MB)")
