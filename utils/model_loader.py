"""
Model loading utilities for TLBVFI.

Extracted from tlbvfi_node.py to enable model reuse across chunk-based nodes.
"""

import torch
import os
import yaml
import argparse
from pathlib import Path
import sys

import folder_paths


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


def load_tlbvfi_model(model_name: str, device: torch.device):
    """
    Load TLBVFI model with VQGAN and UNet weights.

    Args:
        model_name: Name of model file in interpolation folder
        device: torch.device for model placement

    Returns:
        model: Loaded LatentBrownianBridgeModel in eval mode

    Raises:
        FileNotFoundError: If config or model file not found
        ImportError: If TLBVFI model not available
    """
    # Import TLBVFI model
    try:
        current_path = Path(__file__).parent.parent
        tlbvfi_path = current_path / "TLBVFI"
        if tlbvfi_path.is_dir():
            sys.path.insert(0, str(tlbvfi_path))
            from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
        else:
            raise ImportError("TLBVFI directory not found.")
    except ImportError as e:
        raise ImportError(
            f"Could not import TLBVFI model: {e}\n"
            "Please ensure TLBVFI submodule is properly initialized."
        )

    # Load config
    config_path = tlbvfi_path / "configs" / "Template-LBBDM-video.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Make sure the TLBVFI repo is cloned correctly."
        )

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    nconfig = dict2namespace(config)

    # Get model path
    model_path = folder_paths.get_full_path("interpolation", model_name)
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_name} not found in interpolation folder.")

    # Prevent VQGAN from loading checkpoint during initialization
    nconfig.model.VQGAN.params.ckpt_path = None

    # Initialize model structure
    model = LatentBrownianBridgeModel(nconfig.model).to(device)

    # Load weights (same as legacy node - strict=True)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict_to_load = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict_to_load)
    model.eval()

    print(f"TLBVFI: Loaded model {model_name} on {device}")

    return model


def enable_tf32_if_available(device: torch.device) -> bool:
    """
    Enable TF32 acceleration for Ampere+ GPUs (RTX 30/40 series).

    Args:
        device: torch.device to check

    Returns:
        bool: True if TF32 was enabled
    """
    if device.type != 'cuda':
        return False

    # Check if Ampere+ (compute capability >= 8.0)
    if torch.cuda.get_device_capability(device)[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TLBVFI: TF32 acceleration enabled (RTX 30/40 series Tensor Cores)")
        return True

    return False


def enable_cudnn_benchmark(device: torch.device) -> bool:
    """
    Enable cuDNN autotuner for optimal convolution algorithms.

    Args:
        device: torch.device to check

    Returns:
        bool: True if benchmark mode was enabled
    """
    if device.type != 'cuda':
        return False

    torch.backends.cudnn.benchmark = True
    print("TLBVFI: cuDNN autotuner enabled for hardware-specific optimization")
    return True
