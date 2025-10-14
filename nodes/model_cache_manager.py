"""
TLBVFI Model Cache Manager

Provides utility nodes for managing the TLBVFI model cache.
"""

from .tlbvfi_interpolator import clear_model_cache


class TLBVFI_ClearModelCache:
    """
    Clear TLBVFI model cache to free GPU memory.

    Useful when:
    - Switching between different models
    - Running low on VRAM
    - Before loading other heavy models
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "trigger": ("*",),  # Any input to trigger execution
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "clear_cache"
    CATEGORY = "frame_interpolation/TLBVFI-TF32/utils"

    DESCRIPTION = """
Clear TLBVFI model cache and free GPU memory.

📌 Purpose:
- Manually clear cached TLBVFI models from VRAM
- Useful for memory management in complex workflows
- Helps prevent OOM errors when switching models

🎯 Usage:
1. Connect any output to trigger (optional)
2. Execute to clear all cached TLBVFI models
3. Next interpolation will reload the model fresh

⚠️ Note:
- Clearing cache will slow down the next interpolation
- Model will be automatically cached again after reload
- Use only when necessary (switching models, low VRAM, etc.)
    """

    def clear_cache(self, trigger=None):
        """Clear the model cache."""
        clear_model_cache()
        return {"ui": {"text": ["TLBVFI model cache cleared successfully"]}}
