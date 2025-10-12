import torch
import os
import sys
from pathlib import Path
import yaml
import argparse

import folder_paths
from comfy.utils import ProgressBar
import numpy as np
from tqdm import tqdm

# --- Robust Path Handling ---
if 'interpolation' not in folder_paths.folder_names_and_paths:
    new_path = os.path.join(folder_paths.models_dir, 'interpolation')
    os.makedirs(new_path, exist_ok=True)
    folder_paths.folder_names_and_paths['interpolation'] = ([new_path], {'.pth', '.ckpt'})

# --- Helper Functions ---

def dict2namespace(config):
    """Converts a dictionary to a namespace for easier access."""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def find_models(folder_type: str, extensions: list) -> list:
    """Recursively finds all model files with given extensions in the specified folder type."""
    model_list = []
    base_paths = folder_paths.get_folder_paths(folder_type)

    for base_path in base_paths:
        for root, _, files in os.walk(base_path, followlinks=True):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    relative_path = os.path.relpath(os.path.join(root, file), base_path)
                    model_list.append(relative_path.replace("\\", "/"))
    return sorted(list(set(model_list)))

def get_version() -> str:
    """Read version from pyproject.toml."""
    try:
        pyproject_path = Path(__file__).parent / "pyproject.toml"
        if not pyproject_path.exists():
            return "unknown"

        # Try different TOML parsing methods
        try:
            # Python 3.11+ has tomllib built-in
            import tomllib
            with open(pyproject_path, 'rb') as f:
                data = tomllib.load(f)
            return data.get('project', {}).get('version', 'unknown')
        except ImportError:
            pass

        try:
            # Try toml package
            import toml
            with open(pyproject_path, 'r') as f:
                data = toml.load(f)
            return data.get('project', {}).get('version', 'unknown')
        except ImportError:
            pass

        # Fallback: simple regex parsing
        with open(pyproject_path, 'r') as f:
            content = f.read()
            import re
            match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)

        return "unknown"
    except Exception as e:
        print(f"Warning: Could not read version from pyproject.toml: {e}")
        return "unknown"

def calculate_cleanup_interval(device, frame_shape, times_to_interpolate, num_segments):
    """
    Dynamically calculate optimal cleanup interval based on available GPU memory,
    video resolution, and interpolation settings.

    Args:
        device: torch.device (CUDA device)
        frame_shape: tuple (channels, height, width)
        times_to_interpolate: int (interpolation depth)
        num_segments: int (total number of segments to process)

    Returns:
        int: Cleanup interval (number of segments between cache clears)
    """
    if device.type != 'cuda':
        return 10  # Default for CPU

    try:
        # Get GPU memory information
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)

        # Calculate available memory (with safety margin)
        available_memory = total_memory - reserved_memory
        safety_margin = 2 * 1024**3  # 2GB safety margin
        usable_memory = max(available_memory - safety_margin, 1 * 1024**3)

        # Estimate memory per segment
        # Frame: C × H × W × 4 bytes (FP32)
        C, H, W = frame_shape
        frame_size = C * H * W * 4

        # Estimated memory usage per segment:
        # - Input frames: 2 × frame_size
        # - Interpolated frames: (2^times_to_interpolate - 1) × frame_size
        # - Model intermediate: ~10GB (phi_list, encoder/decoder activations)
        # - Overhead: ~1GB
        output_frames_per_seg = 2 ** times_to_interpolate - 1
        frames_memory = (2 + output_frames_per_seg) * frame_size
        model_memory = 10 * 1024**3  # Conservative estimate
        overhead = 1 * 1024**3

        memory_per_segment = frames_memory + model_memory + overhead

        # Calculate how many segments can fit in memory
        segments_before_cleanup = int(usable_memory / memory_per_segment)

        # Apply constraints
        min_interval = 1
        max_interval = 20
        default_interval = 5

        if segments_before_cleanup < min_interval:
            # Critical: need cleanup every segment
            interval = min_interval
            print("=" * 80)
            print("⚠️  TLBVFI WARNING: INSUFFICIENT GPU MEMORY")
            print("=" * 80)
            print(f"GPU Memory: {total_memory / 1024**3:.2f}GB total, {usable_memory / 1024**3:.2f}GB usable")
            print(f"Video: {num_segments} segments, {H}×{W} resolution, t={times_to_interpolate}")
            print(f"Estimated: {memory_per_segment / 1024**3:.2f}GB per segment")
            print(f"")
            print(f"⚠️  Cleanup required EVERY segment to prevent OOM!")
            print(f"⚠️  This will significantly impact performance (~10-20% slower)")
            print(f"⚠️  Consider:")
            print(f"   - Reducing video resolution")
            print(f"   - Using times_to_interpolate=1 instead of {times_to_interpolate}")
            print(f"   - Processing in smaller batches")
            print("=" * 80)
        elif segments_before_cleanup > max_interval:
            # Plenty of memory, use reasonable default
            interval = max_interval
        else:
            # Use calculated interval
            interval = segments_before_cleanup

        # Log the decision
        print(f"TLBVFI: Dynamic cleanup interval = {interval} segments")
        print(f"        (GPU: {usable_memory / 1024**3:.1f}GB usable, "
              f"Est: {memory_per_segment / 1024**3:.1f}GB/segment, "
              f"Video: {H}×{W}, t={times_to_interpolate})")

        return interval

    except Exception as e:
        print(f"TLBVFI Warning: Could not calculate cleanup interval: {e}")
        print(f"TLBVFI: Using default cleanup interval = 5 segments")
        return 5

# --- TLBVFI Setup ---

try:
    current_path = Path(__file__).parent
    tlbvfi_path = current_path / "TLBVFI"
    if tlbvfi_path.is_dir():
        sys.path.insert(0, str(tlbvfi_path))
        from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
    else:
        raise ImportError("TLBVFI directory not found.")
except ImportError as e:
    print("-------------------------------------------------------------------")
    print(f"Error: {e}")
    print("Could not import TLBVFI model from ComfyUI-TLBVFI node.")
    print("Please follow the setup instructions in the README.md file.")
    print("-------------------------------------------------------------------")
    raise

# --- Main Node Class ---

class TLBVFI_VFI_TF32:
    @classmethod
    def INPUT_TYPES(s):
        # We only need the main model file now.
        unet_models = find_models("interpolation", [".pth"])
        if not unet_models:
             raise Exception("No TLBVFI UNet models (.pth) found in 'ComfyUI/models/interpolation/'. Please download 'vimeo_unet.pth'.")

        return {
            "required": {
                "images": ("IMAGE", ),
                "model_name": (unet_models, ),
                "times_to_interpolate": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "gpu_id": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "interpolate"
    CATEGORY = "frame_interpolation/TLBVFI-TF32" # TF32 optimized for RTX 30/40 series

    def interpolate(self, images, model_name, times_to_interpolate, gpu_id):
        # --- Setup ---
        version = get_version()
        print(f"TLBVFI-TF32 v{version} - Video Frame Interpolation")

        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        
        # --- Load Config ---
        tlbvfi_repo_path = Path(__file__).parent / "TLBVFI"
        config_path = tlbvfi_repo_path / "configs" / "Template-LBBDM-video.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}. Make sure the TLBVFI repo is cloned correctly.")
        
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        nconfig = dict2namespace(config)

        # --- Simplified Model Loading ---
        model_path = folder_paths.get_full_path("interpolation", model_name)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_name} not found.")
            
        # Prevent the model from trying to load a VQGAN checkpoint during initialization
        nconfig.model.VQGAN.params.ckpt_path = None
        
        # 1. Initialize the model structure with random weights.
        model = LatentBrownianBridgeModel(nconfig.model).to(device)

        # 2. Load the entire state dict from the single .pth file.
        # This will populate both the VQGAN and the UNet with the correct weights.
        checkpoint = torch.load(model_path, map_location=device)
        
        # The state dict might be nested under a 'model' key.
        state_dict_to_load = checkpoint.get('model', checkpoint)
        
        model.load_state_dict(state_dict_to_load)
        model.eval()

        # GPU Optimizations for RTX 30/40 series (Ampere/Ada architecture)
        if device.type == 'cuda':
            # Enable TF32 for Ampere+ GPUs (RTX 30/40 series)
            # TF32 provides ~8x speedup vs FP32 with same precision, avoiding FP16 dtype issues
            if torch.cuda.get_device_capability(device)[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("TLBVFI: TF32 acceleration enabled (RTX 30/40 series Tensor Cores)")

            # Enable cuDNN autotuner for optimal convolution algorithms
            torch.backends.cudnn.benchmark = True
            print("TLBVFI: cuDNN autotuner enabled for hardware-specific optimization")

        # --- Prepare Images ---
        image_tensors = images.permute(0, 3, 1, 2).float()
        image_tensors = (image_tensors * 2.0) - 1.0
        
        if len(image_tensors) < 2:
            print("TLBVFI Warning: Not enough images to interpolate. Returning original images.")
            return (images, )

        gui_pbar = ProgressBar(len(image_tensors) - 1)

        # Calculate optimal cleanup interval based on GPU memory and video properties
        num_segments = len(image_tensors) - 1
        frame_shape = image_tensors.shape[1:]  # (C, H, W)
        cleanup_interval = calculate_cleanup_interval(
            device=device,
            frame_shape=frame_shape,
            times_to_interpolate=times_to_interpolate,
            num_segments=num_segments
        )

        # Streaming output: collect frames incrementally on CPU to avoid GPU memory explosion
        # This eliminates the need for massive pre-allocated GPU buffer (saves ~9.43GB for t=2)
        output_frames = []

        # Add first frame
        output_frames.append(image_tensors[0])

        # --- Main Interpolation Loop ---
        # Process each segment, immediately transfer results to CPU
        for i in tqdm(range(len(image_tensors) - 1), desc="TLBVFI Interpolating"):
            # Transfer frames to GPU with async copy
            frame1 = image_tensors[i].unsqueeze(0).to(device=device, non_blocking=True)
            frame2 = image_tensors[i+1].unsqueeze(0).to(device=device, non_blocking=True)

            current_frames = [frame1, frame2]
            for _ in range(times_to_interpolate):
                temp_frames = [current_frames[0]]
                for j in range(len(current_frames) - 1):
                    with torch.no_grad():
                        mid_frame = model.sample(current_frames[j], current_frames[j+1], disable_progress=True)
                    temp_frames.extend([mid_frame, current_frames[j+1]])
                current_frames = temp_frames

            # Stream results to CPU immediately (skip first frame as it's already added)
            # Use non_blocking transfer to overlap with next GPU computation
            for frame in current_frames[1:]:
                output_frames.append(frame.squeeze(0).to('cpu', non_blocking=True))

            # Explicit cleanup to prevent memory fragmentation
            del current_frames, temp_frames, frame1, frame2

            # Dynamic memory management based on calculated cleanup interval
            if (i + 1) % cleanup_interval == 0:
                if device.type == 'cuda':
                    torch.cuda.synchronize()  # Wait for all GPU operations to complete
                    torch.cuda.empty_cache()  # Clear memory cache

                    # Print memory usage for monitoring
                    # More frequent for tight memory, less frequent for ample memory
                    monitor_interval = min(50, cleanup_interval * 10)
                    if (i + 1) % monitor_interval == 0:
                        allocated = torch.cuda.memory_allocated(device) / 1024**3
                        reserved = torch.cuda.memory_reserved(device) / 1024**3
                        print(f"TLBVFI: Segment {i+1}/{num_segments} - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

            gui_pbar.update(1)

        # Final GPU memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Stack frames on CPU into final tensor
        print("TLBVFI: Stacking output frames...")
        final_tensors = torch.stack(output_frames, dim=0)
        
        # --- Convert back to ComfyUI's expected format ---
        final_tensors = (final_tensors + 1.0) / 2.0
        final_tensors = final_tensors.clamp(0, 1)
        final_tensors = final_tensors.permute(0, 2, 3, 1)

        return (final_tensors, )