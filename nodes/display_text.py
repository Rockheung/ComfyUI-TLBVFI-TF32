"""
DisplayText - Simple text display node

Displays text output in ComfyUI interface.
"""


class DisplayText:
    """
    Display text in ComfyUI UI.

    Useful as a final node to show results without validation errors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            }
        }

    INPUT_IS_LIST = False
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "display"
    CATEGORY = "frame_interpolation/TLBVFI-TF32/utils"

    DESCRIPTION = """
Display text in ComfyUI interface.

📌 Purpose:
- Shows text output in UI
- Acts as terminal node (no validation errors)
- Useful for displaying stats, paths, or status messages

🎯 Usage:
1. Connect any STRING output (stats, video_path, etc.)
2. Text will appear in node's UI
    """

    def display(self, text):
        """Display text in UI."""
        return {"ui": {"text": [text]}}
