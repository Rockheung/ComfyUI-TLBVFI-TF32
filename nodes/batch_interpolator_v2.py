"""
TLBVFI_BatchInterpolator_V2 - Sequence-wide interpolation helper.

Processes an entire batch of video frames inside a single node by reusing the
production-grade TLBVFI_Interpolator_V2 logic for each adjacent frame pair.
"""

from typing import List, Union

import torch

from .tlbvfi_interpolator_v2 import (
    TLBVFI_Interpolator_V2,
    find_models,
)


class TLBVFI_BatchInterpolator_V2:
    """
    Iterate over a full image batch and interpolate every adjacent frame pair.

    This node is designed for users who prefer to stay within graph wiring
    rather than relying on the ChunkProcessor convenience wrapper. It uses the
    same production-grade interpolator under the hood while returning the full
    sequence (original frames + interpolated frames, or interpolated frames
    only if requested).
    """

    @classmethod
    def INPUT_TYPES(cls):
        unet_models = find_models("interpolation", [".pth"])
        if not unet_models:
            unet_models = ["vimeo_unet.pth (MISSING - please download)"]

        return {
            "required": {
                "images": ("IMAGE",),  # (N, H, W, C) ComfyUI format
                "model_name": (unet_models,),
            },
            "optional": {
                "times_to_interpolate": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 4,
                    "step": 1,
                    "display": "number",
                }),
                "enable_tf32": ("BOOLEAN", {"default": True}),
                "sample_steps": ([10, 20, 50], {"default": 10}),
                "flow_scale": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "cpu_offload": ("BOOLEAN", {"default": True}),
                "gpu_id": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 7,
                    "step": 1,
                }),
                "include_source_frames": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("interpolated_sequence", "frame_count")
    FUNCTION = "interpolate_sequence"
    CATEGORY = "frame_interpolation/TLBVFI-TF32"

    DESCRIPTION = """
Batch interpolate a full video sequence without chunk-based helpers.

🎯 사용 목적:
- 그래프 내에서 직접 배치 연산을 구성하고 싶은 경우
- ChunkProcessor 대신 표준 노드 조합을 선호하는 사용자
- 프레임 전체를 한 번에 처리하면서도 TF32 최적화를 유지

🧠 동작 방식:
- 입력 배치 `images`에서 인접한 모든 프레임 쌍을 순회
- V2 프로덕션 노드를 재사용해 각 쌍을 개별 보간
- 결과를 하나의 연속 시퀀스로 합쳐 반환 (원본 포함 여부 선택 가능)
"""

    def __init__(self):
        # Reuse the production node internally
        self._interpolator = TLBVFI_Interpolator_V2()

    def interpolate_sequence(self, images: torch.Tensor, model_name: str,
                             times_to_interpolate: int = 1, enable_tf32: bool = True,
                             sample_steps: int = 10, flow_scale: float = 0.5,
                             cpu_offload: bool = True, gpu_id: int = 0,
                             include_source_frames: bool = True):
        if not isinstance(images, torch.Tensor):
            raise TypeError(
                "TLBVFI_BatchInterpolator_V2 expects `images` to be a torch.Tensor "
                "in ComfyUI IMAGE format (N, H, W, C)."
            )

        if images.ndim != 4:
            raise ValueError(
                f"Expected `images` tensor with 4 dimensions (N,H,W,C), got shape {tuple(images.shape)}."
            )

        frame_count = images.shape[0]
        if frame_count < 2:
            raise ValueError(
                f"Need at least 2 frames to interpolate, received {frame_count}."
            )

        combined: List[torch.Tensor] = []

        # Iterate over each adjacent pair
        for pair_index in range(frame_count - 1):
            prev_frame = images[pair_index:pair_index + 1]
            next_frame = images[pair_index + 1:pair_index + 2]

            interpolated_tuple = self._interpolator.interpolate(
                prev_frame,
                next_frame,
                model_name,
                times_to_interpolate=times_to_interpolate,
                enable_tf32=enable_tf32,
                sample_steps=sample_steps,
                flow_scale=flow_scale,
                cpu_offload=cpu_offload,
                gpu_id=gpu_id,
            )

            interpolated_frames = interpolated_tuple[0]
            interpolated_frames = interpolated_frames.to("cpu")

            if times_to_interpolate == 0:
                # Single mid-frame
                sequence = torch.cat(
                    [prev_frame.to("cpu"), interpolated_frames, next_frame.to("cpu")],
                    dim=0,
                )
                if include_source_frames:
                    segment = sequence if pair_index == 0 else sequence[1:]
                else:
                    segment = sequence[1:-1] if sequence.shape[0] > 2 else torch.empty(
                        (0,), device=sequence.device
                    )
            else:
                # Recursive branch already includes endpoints
                if include_source_frames:
                    segment = (interpolated_frames
                               if pair_index == 0
                               else interpolated_frames[1:])
                else:
                    inner = interpolated_frames[1:-1]
                    if pair_index == 0:
                        segment = inner
                    else:
                        segment = inner

            if segment.numel() == 0:
                continue

            combined.append(segment)

        if not combined:
            raise RuntimeError("Interpolation produced no frames. Check inputs and settings.")

        output_sequence = torch.cat(combined, dim=0)

        return (output_sequence, output_sequence.shape[0])


NODE_CLASS_MAPPINGS = {
    "TLBVFI_BatchInterpolator_V2": TLBVFI_BatchInterpolator_V2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TLBVFI_BatchInterpolator_V2": "TLBVFI Batch Interpolator V2",
}
