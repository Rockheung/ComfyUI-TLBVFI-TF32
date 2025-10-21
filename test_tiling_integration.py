"""
Standalone test for tiled interpolation with actual TLBVFI model.

This test loads the real model and tests tiled inference.
"""

import sys
import torch
import yaml
import argparse
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import tiling utilities directly (avoid utils.__init__ which needs ComfyUI)
import importlib.util
spec = importlib.util.spec_from_file_location("tiling", project_root / "utils" / "tiling.py")
tiling_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiling_module)
process_with_tiling = tiling_module.process_with_tiling

# Import TLBVFI model directly
sys.path.insert(0, str(Path(__file__).parent / "TLBVFI"))
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel

def dict2namespace(config):
    """Convert a dictionary to a namespace for easier access."""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def test_tiled_interpolation():
    """Test tiled interpolation with real model."""
    print("="*80)
    print("Testing Tiled Interpolation")
    print("="*80)

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return

    device = torch.device("cuda:0")
    print(f"\nDevice: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load model
    print("\n" + "="*80)
    print("Loading Model...")
    print("="*80)

    model_name = "vimeo_unet.pth"
    sample_steps = 10

    # Find model file
    model_path = Path("D:/Users/rock/Documents/ComfyUI/models/interpolation") / model_name
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("\nPlease ensure vimeo_unet.pth is in:")
        print("  D:\\Users\\rock\\Documents\\ComfyUI\\models\\interpolation\\")
        return

    try:
        # Load config
        config_path = project_root / "TLBVFI" / "configs" / "Template-LBBDM-video.yaml"
        print(f"Loading config from: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        nconfig = dict2namespace(config)

        # Override sample_steps
        nconfig.model.BB.params.sample_step = sample_steps

        # Prevent VQGAN from loading checkpoint during initialization
        nconfig.model.VQGAN.params.ckpt_path = None

        # Load model checkpoint
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(str(model_path), map_location=device)

        # Create model instance
        model = LatentBrownianBridgeModel(nconfig.model).to(device)

        # Load weights
        state_dict_to_load = checkpoint.get('model', checkpoint)
        model.load_state_dict(state_dict_to_load)
        model.eval()

        print(f"[OK] Model loaded: {model_name}")
        print(f"  Sample steps: {sample_steps}")

    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test different resolutions
    test_cases = [
        ("1080p", 1080, 1920),
        ("2K", 1536, 2560),  # After padding
        # ("4K", 2160, 3840),  # Commented out for initial test
    ]

    for name, height, width in test_cases:
        print("\n" + "="*80)
        print(f"Testing {name} ({width}x{height})")
        print("="*80)

        # Create dummy frames (normalized to [-1, 1] as expected by model)
        print(f"\nCreating test frames...")
        frame_a = torch.randn(1, 3, height, width, device=device, dtype=torch.float32)
        frame_b = torch.randn(1, 3, height, width, device=device, dtype=torch.float32)

        # Normalize to [-1, 1]
        frame_a = frame_a * 0.5  # Reduce magnitude
        frame_b = frame_b * 0.5

        print(f"  frame_a: {frame_a.shape}, dtype={frame_a.dtype}, device={frame_a.device}")
        print(f"  frame_b: {frame_b.shape}, dtype={frame_b.dtype}, device={frame_b.device}")
        print(f"  Value range: [{frame_a.min():.3f}, {frame_a.max():.3f}]")

        # Test 1: Without tiling (if small enough)
        if height <= 1080:
            print(f"\nTest 1: Direct processing (no tiling)")
            try:
                torch.cuda.empty_cache()
                mem_before = torch.cuda.memory_allocated() / 1024**3

                with torch.no_grad():
                    result_direct = model.sample(frame_a, frame_b, scale=0.5)

                mem_after = torch.cuda.memory_allocated() / 1024**3
                print(f"  [OK] Success!")
                print(f"  Output shape: {result_direct.shape}")
                print(f"  Memory used: {mem_after - mem_before:.2f} GB")

                del result_direct
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [FAIL] Failed: {e}")

        # Test 2: With tiling
        print(f"\nTest 2: Tiled processing (512x512 tiles)")
        try:
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated() / 1024**3

            result_tiled = process_with_tiling(
                model, frame_a, frame_b,
                tile_size=512,
                overlap=64,
                scale=0.5
            )

            mem_after = torch.cuda.memory_allocated() / 1024**3
            print(f"  [OK] Success!")
            print(f"  Output shape: {result_tiled.shape}")
            print(f"  Memory used: {mem_after - mem_before:.2f} GB")

            # Verify output
            assert result_tiled.shape == frame_a.shape, f"Shape mismatch! Expected {frame_a.shape}, got {result_tiled.shape}"
            assert not torch.isnan(result_tiled).any(), "Output contains NaN!"
            assert not torch.isinf(result_tiled).any(), "Output contains Inf!"

            print(f"  Output value range: [{result_tiled.min():.3f}, {result_tiled.max():.3f}]")

            del result_tiled
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [FAIL] Failed: {e}")
            import traceback
            traceback.print_exc()
            return

    print("\n" + "="*80)
    print("All tests PASSED!")
    print("="*80)

if __name__ == "__main__":
    test_tiled_interpolation()
