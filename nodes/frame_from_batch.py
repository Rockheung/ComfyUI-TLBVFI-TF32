"""
TLBVFI_FrameFromBatch - Extract a single frame from an IMAGE batch.

Utility helper that replaces the deprecated GetNode workflow pattern.
"""

from typing import Optional

import torch


class TLBVFI_FrameFromBatch:
    """Extract a specific frame from an IMAGE batch (N, H, W, C)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number",
                }),
                "clamp_index": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frame",)
    FUNCTION = "get_frame"
    CATEGORY = "frame_interpolation/TLBVFI-TF32/utils"

    DESCRIPTION = """
Select a single frame from an IMAGE batch.

🎯 사용 시점:
- FramePairSlicer 출력(배치)을 prev/next 프레임으로 나눌 때
- API 자동화 없이 노드 그래프 안에서 배치 인덱싱이 필요할 때

⚙️ 옵션:
- `index`: 가져올 프레임 번호
- `clamp_index`: true면 범위 밖 인덱스를 자동으로 가장 가까운 값으로 맞춤
"""

    def get_frame(self, images: torch.Tensor, index: int = 0, clamp_index: bool = True):
        if not isinstance(images, torch.Tensor):
            raise TypeError("Expected `images` to be a torch.Tensor in (N,H,W,C) format.")

        if images.ndim != 4:
            raise ValueError(f"`images` must have 4 dimensions (N,H,W,C), got {tuple(images.shape)}.")

        frame_count = images.shape[0]
        if frame_count == 0:
            raise ValueError("Cannot extract frame from empty batch.")

        if clamp_index:
            safe_index = max(0, min(index, frame_count - 1))
        else:
            if not (0 <= index < frame_count):
                raise IndexError(f"Index {index} out of range for batch of size {frame_count}.")
            safe_index = index

        frame = images[safe_index:safe_index + 1]
        return (frame,)


NODE_CLASS_MAPPINGS = {
    "TLBVFI_FrameFromBatch": TLBVFI_FrameFromBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TLBVFI_FrameFromBatch": "TLBVFI Frame From Batch",
}
