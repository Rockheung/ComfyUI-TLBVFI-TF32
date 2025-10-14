"""
TLBVFI_Interpolator Node for Chunk-Based Workflow

Performs frame interpolation on frame pairs with model reuse support.
Refactored from monolithic tlbvfi_node.py for memory-efficient chunk processing.
"""

import torch
import gc
import sys
from pathlib import Path

# Use parent package relative import to avoid conflicts with ComfyUI's utils
try:
    from ..utils import (
        load_tlbvfi_model,
        enable_tf32_if_available,
        enable_cudnn_benchmark,
        cleanup_memory,
        get_memory_stats,
        print_memory_summary,
    )
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import (
        load_tlbvfi_model,
        enable_tf32_if_available,
        enable_cudnn_benchmark,
        cleanup_memory,
        get_memory_stats,
        print_memory_summary,
    )

import folder_paths

# Global model cache to avoid reloading across workflow iterations
_MODEL_CACHE = {}


def find_models(folder_type: str, extensions: list) -> list:
    """Find all model files with given extensions in the specified folder type."""
    import os
    model_list = []
    base_paths = folder_paths.get_folder_paths(folder_type)

    for base_path in base_paths:
        for root, _, files in os.walk(base_path, followlinks=True):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    relative_path = os.path.relpath(os.path.join(root, file), base_path)
                    model_list.append(relative_path.replace("\\", "/"))
    return sorted(list(set(model_list)))


class TLBVFI_Interpolator:
    """
    TLBVFI frame interpolator for chunk-based processing.

    This node performs interpolation on a single frame pair, enabling
    memory-efficient processing of long videos through sequential chunk processing.
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
                "frame_pair": ("IMAGE",),  # (2, H, W, C) from FramePairSlicer
                "model_name": (unet_models,),
                "times_to_interpolate": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "gpu_id": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "is_last_pair": ("BOOLEAN", {"default": False}),  # Include end frame for last pair
                "save_images": ("BOOLEAN", {"default": False}),  # Save frames as PNG for debugging
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("interpolated_frames",)
    FUNCTION = "interpolate"
    CATEGORY = "frame_interpolation/TLBVFI-TF32/chunk"

    DESCRIPTION = """
TLBVFI frame interpolator optimized for chunk-based processing.

📌 Purpose:
- Interpolates between 2 frames using latent diffusion
- Automatic model caching for efficient workflow execution
- Core processing node in chunk-based workflow

🎯 Usage:
1. Connect frame_pair (2 frames) from FramePairSlicer
2. Select model (vimeo_unet.pth)
3. Set times_to_interpolate (1=2x, 2=4x, 3=8x, 4=16x)
4. Connect is_last_pair from FramePairSlicer (ensures last frame is included)
5. [Optional] Set save_images=True to save frames as PNG for debugging

⚡ Features:
- TF32 acceleration on RTX 30/40 series
- Automatic GPU memory management
- Global model caching (models persist across workflow executions)
- Smart frame output: excludes end frame except for last pair
- Debug mode: save frames as PNG images

📊 Output frames:
- Normal pairs: (2^t) frames (e.g., times_to_interpolate=3 → 8 frames)
- Last pair: (2^t + 1) frames (includes final frame of video)

💾 Memory: ~13GB GPU for 4K video

🐛 Debug:
- save_images=True: Saves frames to output/tlbvfi_frames_TIMESTAMP/
    """

    def interpolate(self, frame_pair: torch.Tensor, model_name: str, times_to_interpolate: int,
                   gpu_id: int, is_last_pair: bool = False, save_images: bool = False):
        """
        Interpolate between 2 frames.

        Args:
            frame_pair: (2, H, W, C) tensor from FramePairSlicer
            model_name: Model checkpoint name
            times_to_interpolate: 1=2x, 2=4x, 3=8x, 4=16x
            gpu_id: CUDA device ID
            is_last_pair: If True, includes end frame (for last pair of video)
            save_images: If True, saves frames as PNG for debugging

        Returns:
            interpolated_frames: ((2^t) or (2^t + 1), H, W, C) tensor
                - Normal pairs: 2^t frames (excludes end frame)
                - Last pair: 2^t + 1 frames (includes end frame)
        """
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        # Validation
        if frame_pair.shape[0] != 2:
            raise ValueError(
                f"frame_pair must have exactly 2 frames, got {frame_pair.shape[0]}. "
                f"Use FramePairSlicer to extract frame pairs."
            )

        H, W = frame_pair.shape[1:3]
        print(f"\nTLBVFI_Interpolator: Processing {H}×{W} frame pair with {times_to_interpolate}x interpolation")

        # Model loading with global cache
        cache_key = f"{model_name}_{gpu_id}"
        if cache_key in _MODEL_CACHE:
            model = _MODEL_CACHE[cache_key]
            print(f"TLBVFI_Interpolator: Reusing cached model {model_name}")
        else:
            print(f"TLBVFI_Interpolator: Loading model {model_name}")
            print_memory_summary(device, "Before model load: ")

            model = load_tlbvfi_model(model_name, device)

            # Enable GPU optimizations
            enable_tf32_if_available(device)
            enable_cudnn_benchmark(device)

            # Cache the model for future use
            _MODEL_CACHE[cache_key] = model

            print_memory_summary(device, "After model load: ")
            print(f"TLBVFI_Interpolator: Model cached for future workflow executions")

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
        if save_images:
            import os
            from PIL import Image
            import numpy as np
            from datetime import datetime

            output_dir = folder_paths.get_output_directory()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(output_dir, f"tlbvfi_frames_{timestamp}")
            os.makedirs(save_dir, exist_ok=True)

            print(f"\n  Debug: Saving {len(processed_frames)} frames")
            print(f"  Debug: processed_frames type = {type(processed_frames)}")
            print(f"  Debug: result.shape = {result.shape}")

            for idx, frame_tensor in enumerate(processed_frames):
                print(f"  Debug [{idx}]: frame_tensor type = {type(frame_tensor)}, shape = {frame_tensor.shape}")

                # Convert to numpy uint8
                frame_np = (frame_tensor.numpy() * 255).clip(0, 255).astype(np.uint8)

                print(f"  Debug [{idx}]: frame_np shape = {frame_np.shape}, dtype = {frame_np.dtype}")

                # Ensure correct shape (H, W, C)
                if frame_np.ndim != 3:
                    print(f"  ERROR: Unexpected frame_np dimensions: {frame_np.shape}")
                    continue

                if frame_np.shape[2] != 3:
                    print(f"  ERROR: Expected 3 channels (RGB), got {frame_np.shape[2]}")
                    continue

                img = Image.fromarray(frame_np, mode='RGB')
                img_path = os.path.join(save_dir, f"frame_{idx:04d}.png")
                img.save(img_path)

                if idx == 0:
                    print(f"  Debug: Saved first frame size = {img.size} (width x height)")
                    print(f"  Debug: Image mode = {img.mode}")
                    print(f"  Debug: Sample pixels at (0,0) = {frame_np[0, 0, :]}")
                    print(f"  Debug: Sample pixels at (10,10) = {frame_np[10, 10, :]}")
                    print(f"  Debug: Actual saved file path = {img_path}")

                    # Verify saved image by reading it back
                    saved_img = Image.open(img_path)
                    print(f"  Debug: Read back saved image size = {saved_img.size}")
                    saved_np = np.array(saved_img)
                    print(f"  Debug: Read back numpy shape = {saved_np.shape}")

            print(f"TLBVFI_Interpolator: Saved {len(processed_frames)} frames to {save_dir}")

        # Memory cleanup
        del current_frames, temp_frames, frame1, frame2, image_tensors, processed_frames
        cleanup_memory(device, force_gc=True)

        print_memory_summary(device, "After interpolation: ")
        print(f"TLBVFI_Interpolator: Complete! Generated {result.shape[0]} frames from 2 input frames")
        if not is_last_pair:
            print(f"  (Excluded end frame to avoid duplication in concat)")
        print()

        return (result,)
