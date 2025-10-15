# Frame Interpolation Implementation Overview

**Scope**: Consolidated notes covering the original TLBVFI repository, the ComfyUI-TLBVFI-TF32 node, and reference patterns from ComfyUI-Frame-Interpolation (RIFE/FILM).

---

## 1. Original TLBVFI (Paper Repository)
- Two-stage pipeline: train VQGAN encoder/decoder first, then freeze it while training the Brownian Bridge UNet.
- Brownian Bridge diffusion uses configurable schedules (`linear`, `cosine`) with typical 10-step sampling for fast inference.
- Inference is pairwise (`model.sample(frame_a, frame_b)`), while recursive bisection to reach higher FPS is handled outside the model.
- Adaptive padding ensures spatial dims align with encoder and MaxViT requirements (multiples of 512 for default config).
- Flow-guided decoding exposes a `scale` parameter (0.5 default) to trade detail for speed.

## 2. ComfyUI-TLBVFI-TF32 Node
- Production V2 node reimplements pairwise interpolation, adds adaptive padding, CPU offload, and periodic cache clearing.
- Global model cache keeps the 3.6 GB checkpoint resident per session; cleanup occurs when VRAM pressure is detected.
- TF32 is the optimized path for Ampere/Ada GPUs, providing tensor-core acceleration without precision loss.
- Recursive interpolation now processes one pair at a time, moving completed frames back to CPU immediately to prevent exponential VRAM growth.
- Chunk workflow uses FFmpeg for on-disk storage; roadmap recommends batching pairs and streaming writes to lower subprocess overhead.

## 3. Lessons from RIFE/FILM (ComfyUI-Frame-Interpolation)
- All outputs move to CPU right after inference; RIFE preallocates CPU tensors, FILM appends to lists before concatenation.
- `soft_empty_cache()` runs every N pairs (default 10) plus a final cleanup; both nodes call `gc.collect()` alongside.
- Models reload per workflow run (no global cache); checkpoints cached on disk under `./ckpts/<model_type>/`.
- `torch.no_grad()` consistently wraps FILM forward passes; RIFE relies on `.detach().cpu()`, leaving slight autograd overhead.
- CPU→GPU transfers stay in float32; both projects prioritise VRAM stability over half-precision conversions.

## 4. Consolidated Action Items
1. Expose user override for cache-clear cadence while keeping auto-detected safety defaults.
2. Guard every sampling path with `torch.no_grad()` and verify no auxiliary code re-enables grad.
3. Port adaptive padding/unpadding helpers everywhere frames enter or leave the model.
4. Offer preset sampling schedules (10/20/50 steps; linear/cosine) plus flow-scale control.
5. Streamline FFmpeg usage by keeping long-lived processes or switching to buffered encoders when chunking videos.

---

# 프레임 보간 구현 개요

**범위**: 원본 TLBVFI 저장소, ComfyUI-TLBVFI-TF32 노드, 그리고 ComfyUI-Frame-Interpolation(RIFE/FILM)에서 확인한 패턴을 한 곳에 정리했습니다.

---

## 1. 원본 TLBVFI (논문 저장소)
- 2단계 파이프라인: VQGAN 인코더/디코더를 먼저 학습한 뒤 고정시키고 Brownian Bridge UNet을 학습.
- Brownian Bridge 확산은 `linear`, `cosine` 등 다양한 스케줄을 지원하고, 추론 시 10스텝 샘플링(기본값)으로 속도를 확보.
- 추론 함수는 프레임 쌍(`model.sample(frame_a, frame_b)`)만 처리하며, 높은 FPS를 위한 재귀 분할은 모델 외부에서 수행.
- 인코더/MaxViT 요구사항을 맞추기 위해 입력 크기를 512 배수로 보정하는 적응형 패딩을 사용.
- Flow-guided 디코더는 `scale` 파라미터(기본 0.5)로 디테일과 속도 사이의 트레이드오프를 제어.

## 2. ComfyUI-TLBVFI-TF32 노드
- Production V2 노드는 프레임 쌍 단위 추론을 재구현하고, 적응형 패딩, CPU 오프로드, 주기적 캐시 정리 등을 도입.
- 3.6 GB 체크포인트를 세션 동안 전역 캐시에 유지하며 VRAM 압박 감지 시 정리.
- Ampere/Ada GPU에서는 TF32를 기본 경로로 사용해 텐서 코어 가속과 정밀도를 동시에 확보합니다.
- 재귀 보간은 한 번에 한 쌍만 GPU에서 처리하고 완료된 프레임은 즉시 CPU로 이동해 VRAM 폭증을 방지.
- 청크 기반 워크플로우는 FFmpeg로 디스크에 저장하며, 향후에는 페어 배치 처리와 스트리밍 인코딩으로 프로세스 오버헤드를 줄이는 것이 권장.

## 3. RIFE/FILM에서 가져온 패턴 (ComfyUI-Frame-Interpolation)
- 모든 출력은 추론 직후 CPU로 옮김; RIFE는 CPU 텐서를 미리 할당하고 FILM은 리스트에 담아 마지막에 합치기.
- `soft_empty_cache()`를 기본 10쌍마다 실행하고 마지막에도 호출하며, 항상 `gc.collect()`와 함께 사용.
- 워크플로우 실행마다 모델을 다시 로드하지만 체크포인트는 `./ckpts/<model_type>/` 경로에 디스크 캐시로 저장.
- FILM은 `torch.no_grad()`로 감싸 VRAM 오버헤드를 줄이고, RIFE는 `.detach().cpu()`로 처리해 약간의 오버헤드가 남음.
- CPU→GPU 전송은 float32 기준으로 유지해 VRAM 일관성을 확보합니다.

## 4. 통합 실행 계획
1. 자동 감지된 안전 기본값을 유지하되, 사용자 정의 캐시 정리 주기를 옵션으로 제공.
2. 모든 샘플링 경로에 `torch.no_grad()`가 적용되는지 재점검해 자동 미분 메타데이터가 생성되지 않도록 보장.
3. 입력·출력 경로 전반에 적응형 패딩/언패딩 헬퍼를 이식해 해상도 제약을 제거.
4. 10/20/50 스텝과 linear/cosine 프리셋 등 샘플링 스케줄과 flow-scale 제어를 UI로 노출.
5. 긴 청크 작업 시 FFmpeg 프로세스를 재사용하거나 버퍼 기반 인코더로 전환해 서브프로세스 오버헤드를 최소화.
