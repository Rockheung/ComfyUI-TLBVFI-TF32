# ComfyUI Frame Interpolation: 메모리 관리 & 보간 전략 종합 분석

**분석 대상**: [Fannovel16/ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation)
**분석 일자**: 2025-10-15
**분석자**: Claude Code (CC) + Codex
**비교 대상**: TLBVFI-TF32 구현

---

## 목차

1. [Executive Summary](#executive-summary)
2. [Repository 구조](#repository-구조)
3. [RIFE VFI 심층 분석](#rife-vfi-심층-분석)
4. [FILM VFI 심층 분석](#film-vfi-심층-분석)
5. [메모리 관리 전략 비교표](#메모리-관리-전략-비교표)
6. [ComfyUI 통합 패턴](#comfyui-통합-패턴)
7. [TLBVFI vs RIFE/FILM 비교](#tlbvfi-vs-rifefilm-비교)
8. [Best Practices & 권장사항](#best-practices--권장사항)
9. [실행 가능한 개선 제안](#실행-가능한-개선-제안)

---

## Executive Summary

### 핵심 발견사항

| 측면 | RIFE/FILM | TLBVFI (우리) | 평가 |
|------|-----------|---------------|------|
| **모델 캐싱** | ❌ 매 실행마다 재로드 | ✅ 글로벌 캐시 + 압박 감지 | **TLBVFI 우수** |
| **메모리 정리** | ✅ 주기적 (`soft_empty_cache`) | ⚠️ 수동 (`cleanup_memory`) | **RIFE/FILM 우수** |
| **ComfyUI 통합** | ✅ `model_management` 활용 | ❌ 독립적 구현 | **RIFE/FILM 우수** |
| **중간 프레임 처리** | CPU 즉시 이동 | GPU 누적 후 일괄 전송 | **RIFE/FILM 안전** |
| **autograd 관리** | FILM: ✅, RIFE: ❌ | ✅ `torch.no_grad()` | **TLBVFI/FILM 우수** |
| **비동기 전송** | ❌ 동기 블로킹 | ✅ `non_blocking=True` | **TLBVFI 우수** |

### 주요 교훈

1. **`soft_empty_cache()` 필수**: ComfyUI 생태계와 협조적 메모리 관리
2. **주기적 정리 > 수동 정리**: 긴 워크플로우에서 OOM 방지
3. **CPU 출력 누적**: GPU VRAM 압박 최소화의 기본 패턴
4. **모델 캐싱 트레이드오프**: 단순함 vs 성능 (TLBVFI는 성능 선택이 합리적)

---

## Repository 구조

```
ComfyUI-Frame-Interpolation/
├── vfi_utils.py              # 공통 유틸리티 (핵심!)
├── vfi_models/
│   ├── rife/
│   │   ├── __init__.py       # RIFE 노드 (89-107줄)
│   │   └── rife_arch.py      # IFNet 구조 + backwarp 캐시
│   ├── film/
│   │   ├── __init__.py       # FILM 노드 (63-113줄)
│   │   └── film_arch.py      # TorchScript 모델
│   └── [10+ other VFI models]
├── __init__.py               # NODE_CLASS_MAPPINGS
└── config.yaml               # 체크포인트 경로 설정
```

**핵심 모듈**:
- `vfi_utils.py:124-207`: `_generic_frame_loop` - RIFE의 메모리 관리 핵심
- `vfi_models/film/__init__.py:12-42`: FILM의 적응형 샘플링 알고리즘
- `comfy.model_management`: `get_torch_device()`, `soft_empty_cache()`

---

## RIFE VFI 심층 분석

### 1. 모델 로딩 & 캐싱 전략

```python
# vfi_models/rife/__init__.py:89-94
from .rife_arch import IFNet
model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)  # 디스크 캐시 활용
arch_ver = CKPT_NAME_VER_DICT[ckpt_name]
interpolation_model = IFNet(arch_ver=arch_ver)
interpolation_model.load_state_dict(torch.load(model_path))
interpolation_model.eval().to(get_torch_device())  # ComfyUI 디바이스 존중
```

**특징**:
- ❌ **인-메모리 캐시 없음**: 매 워크플로우 실행마다 Python 재인스턴스화
- ✅ **디스크 캐시**: `./ckpts/rife/` 아래에 체크포인트 보관 (재다운로드 방지)
- ✅ **ComfyUI 통합**: `get_torch_device()`로 글로벌 디바이스 선택 따름
- 📦 **버전 관리**: `CKPT_NAME_VER_DICT`로 아키텍처 버전 매핑 (4.0~4.9)

### 2. 프레임 처리 파이프라인

```
입력: IMAGE tensor (N, H, W, C)
  ↓
[preprocess_frames] → (N, C, H, W) CPU tensor
  ↓
[generic_frame_loop] ← 핵심 처리 루프
  ├─ 사전 할당: torch.zeros(multiplier*N, C, H, W, device="cpu")
  ├─ For each pair (i, i+1):
  │   ├─ frame0, frame1 → GPU (float32)
  │   ├─ For timestep in [1/m, 2/m, ..., (m-1)/m]:
  │   │   └─ IFNet(frame0, frame1, timestep) → mid_frame
  │   │   └─ mid_frame.detach().cpu() → 즉시 CPU 이동
  │   ├─ 카운터 증가
  │   └─ If counter >= clear_cache_after_n_frames:
  │       ├─ soft_empty_cache()  # ComfyUI 메모리 협상
  │       ├─ gc.collect()
  │       └─ counter = 0
  └─ soft_empty_cache()  # 최종 정리
  ↓
[postprocess_frames] → (N, H, W, C) 출력
```

### 3. 메모리 관리 핵심 메커니즘

#### A. CPU 사전 할당 전략
```python
# vfi_utils.py:147
output_frames = torch.zeros(
    multiplier * frames.shape[0],
    *frames.shape[1:],
    dtype=dtype,  # float16 (메모리 절약)
    device="cpu"  # GPU VRAM 압박 방지
)
```

**장점**:
- 출력 크기를 미리 알 수 있으므로 효율적
- GPU VRAM 사용 최소화 (현재 페어 + 모델만)

**트레이드오프**:
- 큰 비디오는 RAM 사전 할당이 클 수 있음
- FILM은 동적 리스트 사용 (유연하지만 concat 오버헤드)

#### B. 즉시 CPU 전송
```python
# vfi_utils.py:170-176
middle_frame = return_middle_frame_function(
    frame0.to(DEVICE),
    frame1.to(DEVICE),
    timestep,
    *args
).detach().cpu()  # ← 핵심: 즉시 CPU로, gradient 끊기
```

**중요**: RIFE는 `torch.no_grad()`를 사용하지 않음
- `.detach()` 전에 autograd 메타데이터가 잠깐 생성됨
- FILM/TLBVFI보다 약간 더 높은 VRAM 오버헤드

#### C. 주기적 캐시 클리어 (사용자 설정 가능)
```python
# vfi_utils.py:186-192
if number_of_frames_processed >= clear_cache_after_n_frames:
    print("Comfy-VFI: Clearing cache...", end=' ')
    soft_empty_cache()  # ComfyUI의 스마트 캐시 관리
    number_of_frames_processed = 0
    print("Done cache clearing")
gc.collect()  # 매 반복마다 Python GC
```

**`soft_empty_cache()` vs `torch.cuda.empty_cache()`**:
- `soft_empty_cache()`: ComfyUI의 다른 노드와 메모리 협상
- 단순 `empty_cache()`: 모든 CUDA 캐시 강제 해제 (다른 워크플로우 영향)

#### D. backwarp 그리드 캐싱
```python
# vfi_models/rife/rife_arch.py:16-70
backwarp_tenGrid = {}  # 글로벌 딕셔너리

def warp(tenIn, tenFlow):
    if str(tenFlow.device) + str(tenFlow.shape) not in backwarp_tenGrid:
        # 샘플링 그리드 계산 & 캐싱
        backwarp_tenGrid[str(tenFlow.device) + str(tenFlow.shape)] = ...
    return torch.nn.functional.grid_sample(...)
```

**효과**: 동일 해상도 처리 시 그리드 재계산 방지 (소폭 성능 향상)

---

## FILM VFI 심층 분석

### 1. 모델 로딩 전략

```python
# vfi_models/film/__init__.py:73-76
model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
model = torch.jit.load(model_path, map_location='cpu')  # TorchScript
model.eval()
model = model.to(DEVICE)  # CPU 먼저 로드 후 GPU 이동 (안전)
```

**차별점**:
- **TorchScript 패키징**: 단일 `.pt` 파일로 배포 (아키텍처 코드 불필요)
- **CPU 먼저 로드**: GPU 스파이크 방지 (큰 모델에 유리)
- ❌ **캐싱 없음**: RIFE와 동일하게 매번 재로드

### 2. 적응형 재귀적 보간 알고리즘

FILM의 핵심 차별화 요소:

```python
# vfi_models/film/__init__.py:12-42
def inference(model, img_batch_1, img_batch_2, inter_frames):
    results = [img_batch_1, img_batch_2]
    idxes = [0, inter_frames + 1]
    remains = list(range(1, inter_frames + 1))
    splits = torch.linspace(0, 1, inter_frames + 2)  # 시간 분할

    for _ in range(len(remains)):
        # 1. 거리 행렬 계산: 어떤 프레임을 다음에 생성할지 결정
        starts = splits[idxes[:-1]]
        ends = splits[idxes[1:]]
        distances = ((splits[None, remains] - starts[:, None]) /
                     (ends[:, None] - starts[:, None]) - .5).abs()

        # 2. 가장 중앙에 가까운 위치 선택
        matrix = torch.argmin(distances).item()
        start_i, step = np.unravel_index(matrix, distances.shape)
        end_i = start_i + 1

        # 3. 동적 timestep 계산
        x0 = results[start_i].to(DEVICE)
        x1 = results[end_i].to(DEVICE)
        dt = (splits[remains[step]] - splits[idxes[start_i]]) /
             (splits[idxes[end_i]] - splits[idxes[start_i]])

        # 4. 추론 & 삽입
        with torch.no_grad():  # ← RIFE와 달리 명시적 no_grad
            prediction = model(x0, x1, dt)

        insert_position = bisect.bisect_left(idxes, remains[step])
        idxes.insert(insert_position, remains[step])
        results.insert(insert_position, prediction.clamp(0, 1).float())
        del remains[step]

    return [tensor.flip(0) for tensor in results]
```

**알고리즘 특징**:
- 🎯 **적응형 샘플링**: 시간적으로 균등 분배 (중앙부터 생성)
- 🔄 **재귀적**: 이미 생성된 프레임 사이에 새 프레임 삽입
- 📊 **동적 timestep**: 프레임 간 거리 기반 계산
- 💾 **메모리 효율**: 한 번에 2개 프레임만 GPU

**RIFE와의 차이**:
- RIFE: 순차적 타임스텝 (1/m, 2/m, ..., (m-1)/m)
- FILM: 거리 최소화 기반 적응형 선택

### 3. 메모리 관리 (RIFE와 차이점)

```python
# vfi_models/film/__init__.py:79-113
output_frames = []  # 동적 리스트 (사전 할당 X)

for frame_itr in range(len(frames) - 1):
    frame_0 = frames[frame_itr:frame_itr+1].to(DEVICE).float()
    frame_1 = frames[frame_itr+1:frame_itr+2].to(DEVICE).float()

    result = inference(model, frame_0, frame_1, multipliers[frame_itr] - 1)

    # CPU 이동 & 리스트 누적
    output_frames.extend([
        frame.detach().cpu().to(dtype=dtype)
        for frame in result[:-1]  # 마지막 프레임 제외 (중복 방지)
    ])

    # 동일한 주기적 캐시 클리어
    if processed_count >= clear_cache_after_n_frames:
        soft_empty_cache()
        gc.collect()

output_frames.append(frames[-1:])
out = torch.cat(output_frames, dim=0)  # 최종 concat
soft_empty_cache()  # 최종 정리
```

**RIFE와 비교**:
| 측면 | RIFE | FILM |
|------|------|------|
| **출력 할당** | 사전 할당 `torch.zeros` | 동적 `list.extend` |
| **메모리 예측** | 쉬움 (크기 고정) | 어려움 (가변) |
| **유연성** | 낮음 | 높음 (per-pair multiplier) |
| **concat 오버헤드** | 없음 | 있음 (최종 `torch.cat`) |

---

## 메모리 관리 전략 비교표

### 공통 패턴

| 전략 | RIFE | FILM | TLBVFI | 평가 |
|------|------|------|--------|------|
| **CPU 출력 누적** | ✅ 사전 할당 | ✅ 동적 리스트 | ✅ 최종만 | **공통 Best Practice** |
| **detach() 사용** | ✅ | ✅ | ✅ | **필수 패턴** |
| **주기적 정리** | ✅ 10프레임 | ✅ 10프레임 | ❌ 없음 | **RIFE/FILM 우수** |
| **gc.collect()** | ✅ 매 반복 | ✅ 매 반복 | ✅ 수동 | **공통** |
| **최종 정리** | ✅ soft_empty | ✅ soft_empty | ✅ cleanup_memory | **RIFE/FILM 우수** |

### 차별화 요소

| 측면 | RIFE | FILM | TLBVFI |
|------|------|------|--------|
| **모델 캐싱** | ❌ 매번 로드 | ❌ 매번 로드 | ✅ 글로벌 캐시 |
| **autograd** | ❌ detach만 | ✅ no_grad | ✅ no_grad |
| **비동기 전송** | ❌ 동기 | ❌ 동기 | ✅ non_blocking |
| **출력 dtype** | float16 | float32 | float32 |
| **모델 타입** | 표준 PyTorch | TorchScript | 표준 PyTorch |
| **메모리 압박 감지** | ❌ | ❌ | ✅ 자동 체크 |

---

## ComfyUI 통합 패턴

### RIFE/FILM의 통합 방식

```python
from comfy.model_management import get_torch_device, soft_empty_cache

DEVICE = get_torch_device()  # 글로벌 디바이스 선택 존중

# 노드 등록
NODE_CLASS_MAPPINGS = {
    "RIFE VFI": RIFE_VFI,
    "FILM VFI": FILM_VFI,
}
```

**특징**:
- ✅ **최소 침입적**: 디바이스 선택 & 캐시 관리만 활용
- ✅ **표준 패턴**: ComfyUI의 다른 VFI 노드와 일관성
- ❌ **고급 기능 미사용**: `LoadedModel`, `model_management` 레지스트리 등

### Frame Skipping (InterpolationStateList)

```python
# vfi_utils.py:27-59
class InterpolationStateList:
    def __init__(self, frame_indices: List[int], is_skip_list: bool):
        self.frame_indices = frame_indices
        self.is_skip_list = is_skip_list

    def is_frame_skipped(self, frame_index):
        is_frame_in_list = frame_index in self.frame_indices
        return (self.is_skip_list and is_frame_in_list) or
               (not self.is_skip_list and not is_frame_in_list)
```

**사용 예**:
```python
# 워크플로우에서
states = InterpolationStateList([10, 20, 30], is_skip_list=True)
# → 10, 20, 30번 프레임은 보간 스킵
```

**장점**: 워크플로우 주도적 제어 (노드 상태 없음)

---

## TLBVFI vs RIFE/FILM 비교

### 코드 비교

#### TLBVFI (현재 구현)
```python
# nodes/tlbvfi_interpolator.py
_MODEL_CACHE = {}  # 글로벌 캐싱

def interpolate(self, frame_pair, model_name, times_to_interpolate, gpu_id, ...):
    device = torch.device(f"cuda:{gpu_id}")

    # 모델 로딩 with 캐싱 & 압박 감지
    cache_key = f"{model_name}_{gpu_id}"
    if cache_key in _MODEL_CACHE:
        model = _MODEL_CACHE[cache_key]
    else:
        # 메모리 압박 자동 감지
        if device.type == 'cuda':
            mem_stats = get_memory_stats(device)
            if mem_stats['free'] < 4.0:
                clear_model_cache()  # 자동 정리

        model = load_tlbvfi_model(model_name, device)
        _MODEL_CACHE[cache_key] = model

    # GPU에서 모든 반복 보간 처리
    current_frames = [frame1, frame2]
    for iteration in range(times_to_interpolate):
        temp_frames = [current_frames[0]]
        for j in range(len(current_frames) - 1):
            with torch.no_grad():
                mid_frame = model.sample(current_frames[j], current_frames[j+1])
            temp_frames.extend([mid_frame, current_frames[j+1]])
        current_frames = temp_frames

    # 후처리: GPU → CPU (최종만)
    processed_frames = []
    for frame in frames_to_process:
        frame_cpu = frame.squeeze(0).to('cpu', non_blocking=True)  # 비동기
        frame_cpu = (frame_cpu + 1.0) / 2.0
        frame_cpu = frame_cpu.clamp(0, 1)
        processed_frames.append(frame_cpu.permute(1, 2, 0))

    result = torch.stack(processed_frames, dim=0)

    # 정리
    del current_frames, temp_frames, frame1, frame2
    cleanup_memory(device, force_gc=True)  # 수동 정리

    return (result,)
```

#### RIFE/FILM 패턴
```python
# 매번 로드
model = load_model(model_path)
model.eval().to(device)

# 프레임마다 GPU → CPU 즉시 전송
for pair in pairs:
    result = model(pair).detach().cpu()  # 즉시
    output.append(result)

    if count >= 10:
        soft_empty_cache()  # 주기적
        count = 0
```

### 장단점 비교표

| 측면 | TLBVFI | RIFE/FILM | 승자 |
|------|--------|-----------|------|
| **재실행 속도** | ⚡ 즉시 (캐싱) | 🐌 3-5초 로드 | **TLBVFI** |
| **메모리 안전성** | ✅ 자동 감지 | ✅ 주기적 정리 | **동점** |
| **ComfyUI 협조** | ❌ 독립적 | ✅ soft_empty_cache | **RIFE/FILM** |
| **GPU 중간 누적** | ⚠️ 반복 시 압박 | ✅ 즉시 CPU | **RIFE/FILM** |
| **비동기 최적화** | ✅ non_blocking | ❌ 동기 | **TLBVFI** |
| **단순성** | ❌ 복잡 (캐싱 로직) | ✅ 단순 | **RIFE/FILM** |
| **autograd 관리** | ✅ no_grad | RIFE: ❌, FILM: ✅ | **TLBVFI/FILM** |

### Codex의 추가 발견

- **적응형 정리 주기**: `calculate_cleanup_interval`이 실시간 VRAM 통계를 바탕으로 정리 주기를 조정합니다. RIFE/FILM이 고정 10프레임 주기를 사용하는 것과 대비되는 차별점입니다.
- **배치 스트리밍 파이프라인**: TLBVFI는 non-blocking CPU 전송으로 프레임 배치를 스트리밍 처리하며, 사용 가능한 RAM을 기준으로 배치 크기를 동적으로 조정합니다. (`tlbvfi_node.py:182-215`)
- **명시적 모델 언로드**: 실행 종료 시 모델을 CPU로 되돌리고 `torch.cuda.empty_cache()`를 호출해 VRAM을 즉시 회수합니다. (`tlbvfi_node.py:431-482`)
- **수동 캐시 간격 노출 필요**: Codex 분석에 따르면 고급 사용자가 VRAM 여건에 맞춰 정리 간격을 직접 조정할 수 있는 옵션을 제공하면 운영 안정성이 높아집니다.

---

## Best Practices & 권장사항

### 1. ComfyUI 통합 개선 (높음 우선순위)

**현재 TLBVFI**:
```python
cleanup_memory(device, force_gc=True)
```

**권장**:
```python
from comfy.model_management import soft_empty_cache

# 주기적 정리 (ChunkProcessor에 추가)
if (pair_idx + 1) % 10 == 0:
    soft_empty_cache()  # ComfyUI와 협조
    gc.collect()

# 최종 정리
soft_empty_cache()
```

**이유**:
- ComfyUI의 다른 노드와 메모리 협상
- 멀티 워크플로우 환경에서 안전
- 표준 패턴 준수

### 2. torch.no_grad() 보장 (높음)

**현재**: 이미 구현됨 ✅
```python
with torch.no_grad():
    mid_frame = model.sample(...)
```

**확인 필요**: 모든 추론 경로에 적용되었는지 검증

### 3. 사용자 설정 가능한 캐시 클리어 (중간)

```python
class TLBVFI_ChunkProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                ...
                "clear_cache_after_n_pairs": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "tooltip": "주기적 캐시 클리어 빈도 (낮을수록 안전, 느림)"
                }),
            }
        }

    def process_all_chunks(self, ..., clear_cache_after_n_pairs=10):
        for pair_idx in range(total_pairs):
            ...
            if (pair_idx + 1) % clear_cache_after_n_pairs == 0:
                soft_empty_cache()
```

**효과**: 사용자가 메모리/속도 트레이드오프 제어

### 4. 메모리 프로파일링 추가 (낮음)

```python
def process_all_chunks(self, ...):
    for pair_idx in range(total_pairs):
        if pair_idx % 50 == 0:  # 50개마다 로깅
            mem_stats = get_memory_stats(device)
            print(f"[Pair {pair_idx}/{total_pairs}] "
                  f"VRAM: {mem_stats['used']:.1f}/{mem_stats['total']:.1f} GB")
```

### 5. TorchScript 패키징 고려 (선택)

**Codex 제안** (line 134-141):
> FILM의 TorchScript 패키징은 로딩을 단순화하고 CPU 친화적.
> TLBVFI의 추론 경로를 TorchScript로 export하면 향후 로드 속도 개선 가능.

**구현 난이도**: 높음 (TLBVFI 모델 구조 복잡도 고려)

### 6. dtype 최적화 (선택)

현재 TLBVFI는 `float32`, RIFE는 `float16` 사용.

**옵션 A: 출력만 float16**
```python
processed_frames = []
for frame in frames_to_process:
    frame_cpu = frame.squeeze(0).to('cpu', non_blocking=True)
    frame_cpu = (frame_cpu + 1.0) / 2.0
    frame_cpu = frame_cpu.clamp(0, 1).to(dtype=torch.float16)  # 저장 시 변환
    processed_frames.append(frame_cpu.permute(1, 2, 0))
```

**효과**: 메모리 절약 (실질적 정확도 손실 거의 없음)

---

## 실행 가능한 개선 제안

### Phase 1: 즉시 적용 가능 (1-2시간)

1. **ComfyUI 통합**
   ```python
   # utils/memory_manager.py에 추가
   def soft_empty_cache_wrapper():
       """ComfyUI의 soft_empty_cache를 사용하되, 없으면 fallback"""
       try:
           from comfy.model_management import soft_empty_cache
           soft_empty_cache()
       except ImportError:
           torch.cuda.empty_cache()
       gc.collect()
   ```

2. **ChunkProcessor에 주기적 정리**
   ```python
   # nodes/chunk_processor.py:240-280
   for pair_idx in tqdm(range(total_pairs)):
       ...
       if (pair_idx + 1) % 10 == 0:
           from utils.memory_manager import soft_empty_cache_wrapper
           soft_empty_cache_wrapper()
   ```

### Phase 2: 사용성 개선 (2-4시간)

3. **사용자 설정 추가**
   - `clear_cache_after_n_pairs` 파라미터
   - 기본값 10, 툴팁으로 설명

4. **메모리 프로파일링 로깅**
   - 50 페어마다 VRAM 사용량 출력
   - 디버깅 시 유용

### Phase 3: 최적화 (선택적)

5. **dtype 최적화**
   - 출력 float16 변환 테스트
   - 품질 비교 후 결정

6. **TorchScript 조사**
   - TLBVFI 모델의 script 가능성 검토
   - ROI 평가 후 진행 여부 결정

---

## 결론

### 핵심 Takeaways

1. **RIFE/FILM은 단순함을 통한 안정성**
   - 모델 재로드 오버헤드를 감수
   - 주기적 정리로 OOM 방지
   - ComfyUI 표준 패턴 준수

2. **TLBVFI는 성능을 위한 복잡성**
   - 모델 캐싱으로 재실행 속도 향상
   - 메모리 압박 자동 감지
   - 비동기 전송 최적화

3. **두 접근의 하이브리드가 이상적**
   - TLBVFI의 캐싱 + RIFE/FILM의 주기적 정리
   - 성능과 안정성 모두 확보

### 다음 단계

1. ✅ **즉시 적용**: `soft_empty_cache()` 통합
2. ⏰ **단기**: 주기적 캐시 클리어 추가
3. 📊 **중기**: 사용자 설정 & 프로파일링
4. 🔬 **장기**: TorchScript 패키징 조사

---

## 참고자료

- **소스 코드**: [Fannovel16/ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation)
- **Codex 분석 요약**: 본문 「Codex의 추가 발견」 절에 통합 반영
- **ComfyUI 문서**: [docs.comfy.org](https://docs.comfy.org)
- **RIFE Paper**: [Real-Time Intermediate Flow Estimation (ECCV 2022)](https://arxiv.org/abs/2011.06294)
- **FILM Paper**: [Frame Interpolation for Large Motion (ECCV 2022)](https://arxiv.org/abs/2202.04901)
- **ComfyUI Model Management**: [`comfy/model_management.py`](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_management.py)

### 주요 파일 참조

| 파일 | 라인 | 내용 |
|------|------|------|
| `vfi_utils.py` | 124-207 | `_generic_frame_loop` 핵심 로직 |
| `vfi_models/rife/__init__.py` | 89-107 | RIFE 노드 구현 |
| `vfi_models/film/__init__.py` | 12-42, 63-113 | FILM 알고리즘 & 노드 |
| `vfi_models/rife/rife_arch.py` | 16-70 | backwarp 그리드 캐싱 |
| `tlbvfi_node.py` | 86-177, 304-479 | TLBVFI 적응형 정리 & 언로드 |

---

**문서 버전**: 2.0 (CC + Codex 통합)
**최종 업데이트**: 2025-10-15
**상태**: ✅ 실행 준비 완료
