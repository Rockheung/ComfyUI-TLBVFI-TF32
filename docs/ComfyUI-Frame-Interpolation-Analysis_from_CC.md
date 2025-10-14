# ComfyUI-Frame-Interpolation 메모리 관리 및 프레임 보간 전략 분석

**분석 대상**: [Fannovel16/ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation)
**분석 일자**: 2025-10-15
**분석자**: Claude Code + Rockheung

---

## 목차

1. [개요](#개요)
2. [RIFE VFI 구현 분석](#rife-vfi-구현-분석)
3. [FILM VFI 구현 분석](#film-vfi-구현-분석)
4. [메모리 관리 전략 비교](#메모리-관리-전략-비교)
5. [Best Practices 추출](#best-practices-추출)
6. [TLBVFI 구현과의 비교](#tlbvfi-구현과의-비교)
7. [권장사항](#권장사항)

---

## 개요

ComfyUI-Frame-Interpolation은 10가지 이상의 VFI 모델(RIFE, FILM, AMT, FLAVR 등)을 제공하는 종합 프레임 보간 확장입니다. 이 분석에서는 특히 RIFE와 FILM의 메모리 관리 및 프레임 처리 전략에 집중합니다.

### 주요 발견사항

- **모델 캐싱 없음**: 매 실행마다 모델을 새로 로드
- **주기적 캐시 클리어**: 기본 10프레임마다 CUDA 캐시 정리
- **CPU 기반 출력 누적**: GPU가 아닌 CPU에 결과 누적
- **ComfyUI 통합**: `comfy.model_management`의 `soft_empty_cache()` 활용

---

## RIFE VFI 구현 분석

### 파일 위치
- **노드 구현**: `/vfi_models/rife/__init__.py`
- **아키텍처**: `/vfi_models/rife/rife_arch.py`
- **공통 유틸리티**: `/vfi_utils.py`

### 모델 로딩 전략

```python
# vfi_models/rife/__init__.py:89-94
from .rife_arch import IFNet
model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
arch_ver = CKPT_NAME_VER_DICT[ckpt_name]
interpolation_model = IFNet(arch_ver=arch_ver)
interpolation_model.load_state_dict(torch.load(model_path))
interpolation_model.eval().to(get_torch_device())
```

**특징**:
- ✅ **매 실행마다 모델 로드**: 캐싱 없음
- ✅ **즉시 GPU로 이동**: `get_torch_device()`로 자동 디바이스 선택
- ✅ **eval 모드 설정**: 추론 최적화
- ❌ **재사용 없음**: 다음 실행 시 메모리에서 해제

### 프레임 처리 루프

RIFE는 `generic_frame_loop()`를 사용:

```python
# vfi_utils.py:124-207 (_generic_frame_loop 함수)
def _generic_frame_loop(
        frames,
        clear_cache_after_n_frames,
        multiplier,
        return_middle_frame_function,
        *args,
        use_timestep=True,
        dtype=torch.float16):

    # CPU에 출력 프레임 사전 할당
    output_frames = torch.zeros(multiplier*frames.shape[0], *frames.shape[1:],
                                 dtype=dtype, device="cpu")
    out_len = 0
    number_of_frames_processed_since_last_cleared_cuda_cache = 0

    for frame_itr in range(len(frames) - 1):
        frame0 = frames[frame_itr:frame_itr+1]
        output_frames[out_len] = frame0  # 첫 프레임 복사
        out_len += 1

        frame0 = frame0.to(dtype=torch.float32)
        frame1 = frames[frame_itr+1:frame_itr+2].to(dtype=torch.float32)

        # 중간 프레임 생성
        for middle_i in range(1, multiplier):
            timestep = middle_i/multiplier

            middle_frame = return_middle_frame_function(
                frame0.to(DEVICE),
                frame1.to(DEVICE),
                timestep,
                *args
            ).detach().cpu()  # 즉시 CPU로 이동

            middle_frame_batches.append(middle_frame.to(dtype=dtype))

        # 출력에 복사
        for middle_frame in middle_frame_batches:
            output_frames[out_len] = middle_frame
            out_len += 1

        number_of_frames_processed_since_last_cleared_cuda_cache += 1

        # 주기적 캐시 클리어
        if number_of_frames_processed_since_last_cleared_cuda_cache >= clear_cache_after_n_frames:
            print("Comfy-VFI: Clearing cache...", end=' ')
            soft_empty_cache()
            number_of_frames_processed_since_last_cleared_cuda_cache = 0
            print("Done cache clearing")

        gc.collect()

    # 마지막 프레임 추가
    output_frames[out_len] = frames[-1:]
    out_len += 1

    # 최종 캐시 클리어
    soft_empty_cache()

    return output_frames[:out_len]
```

### 메모리 관리 핵심 전략

1. **사전 할당 CPU 텐서**
   ```python
   output_frames = torch.zeros(multiplier*frames.shape[0], *frames.shape[1:],
                                dtype=dtype, device="cpu")
   ```
   - 모든 출력을 CPU 메모리에 미리 할당
   - GPU VRAM 사용 최소화

2. **즉시 CPU 전송**
   ```python
   middle_frame = return_middle_frame_function(...).detach().cpu()
   ```
   - 생성된 프레임을 즉시 CPU로 이동
   - `.detach()`로 gradient 추적 제거

3. **주기적 캐시 클리어**
   ```python
   if processed_frames >= clear_cache_after_n_frames:
       soft_empty_cache()  # ComfyUI의 스마트 캐시 클리어
       gc.collect()         # Python GC
   ```
   - 기본 10프레임마다
   - 사용자 설정 가능 (`clear_cache_after_n_frames` 파라미터)

4. **최종 정리**
   ```python
   soft_empty_cache()  # 작업 완료 후 전체 정리
   ```

---

## FILM VFI 구현 분석

### 파일 위치
- **노드 구현**: `/vfi_models/film/__init__.py`
- **아키텍처**: TorchScript 모델 (`torch.jit.load`)

### 모델 로딩 전략

```python
# vfi_models/film/__init__.py:73-76
model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
model = torch.jit.load(model_path, map_location='cpu')
model.eval()
model = model.to(DEVICE)
```

**특징**:
- ✅ **TorchScript 모델**: JIT 컴파일된 모델 사용
- ✅ **CPU 먼저 로드**: `map_location='cpu'`로 안전하게 로드
- ✅ **명시적 디바이스 이동**: 이후 `to(DEVICE)`
- ❌ **캐싱 없음**: RIFE와 동일하게 매번 로드

### 프레임 처리 루프

FILM은 **독자적인 처리 루프** 사용:

```python
# vfi_models/film/__init__.py:79-113
frames = preprocess_frames(frames)
number_of_frames_processed_since_last_cleared_cuda_cache = 0
output_frames = []  # 리스트로 동적 누적

for frame_itr in range(len(frames) - 1):
    # GPU로 프레임 전송
    frame_0 = frames[frame_itr:frame_itr+1].to(DEVICE).float()
    frame_1 = frames[frame_itr+1:frame_itr+2].to(DEVICE).float()

    # FILM의 재귀적 inference
    result = inference(model, frame_0, frame_1, multipliers[frame_itr] - 1)

    # CPU로 즉시 이동 후 리스트에 추가
    output_frames.extend([
        frame.detach().cpu().to(dtype=dtype)
        for frame in result[:-1]
    ])

    number_of_frames_processed_since_last_cleared_cuda_cache += 1

    # 주기적 캐시 클리어
    if number_of_frames_processed_since_last_cleared_cuda_cache >= clear_cache_after_n_frames:
        print("Comfy-VFI: Clearing cache...", end = ' ')
        soft_empty_cache()
        number_of_frames_processed_since_last_cleared_cuda_cache = 0
        print("Done cache clearing")

    gc.collect()

output_frames.append(frames[-1:].to(dtype=dtype))
output_frames = [frame.cpu() for frame in output_frames]  # 최종 CPU 확인
out = torch.cat(output_frames, dim=0)  # 한번에 concat

# 최종 캐시 클리어
soft_empty_cache()
return (postprocess_frames(out), )
```

### FILM의 특수한 Inference 전략

```python
# vfi_models/film/__init__.py:12-42
def inference(model, img_batch_1, img_batch_2, inter_frames):
    results = [img_batch_1, img_batch_2]
    idxes = [0, inter_frames + 1]
    remains = list(range(1, inter_frames + 1))
    splits = torch.linspace(0, 1, inter_frames + 2)

    for _ in range(len(remains)):
        # 거리 기반으로 다음 프레임 선택
        starts = splits[idxes[:-1]]
        ends = splits[idxes[1:]]
        distances = ((splits[None, remains] - starts[:, None]) /
                     (ends[:, None] - starts[:, None]) - .5).abs()

        matrix = torch.argmin(distances).item()
        start_i, step = np.unravel_index(matrix, distances.shape)
        end_i = start_i + 1

        x0 = results[start_i].to(DEVICE)
        x1 = results[end_i].to(DEVICE)
        dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) /
             (splits[idxes[end_i]] - splits[idxes[start_i]])

        with torch.no_grad():
            prediction = model(x0, x1, dt)

        insert_position = bisect.bisect_left(idxes, remains[step])
        idxes.insert(insert_position, remains[step])
        results.insert(insert_position, prediction.clamp(0, 1).float())
        del remains[step]

    return [tensor.flip(0) for tensor in results]
```

**특징**:
- 🎯 **적응형 샘플링**: 시간적으로 균등하게 분배된 프레임 생성
- 🔄 **재귀적 보간**: 이미 생성된 프레임 사이에 새 프레임 삽입
- 📊 **동적 타임스텝**: 프레임 간 거리 기반으로 timestep 계산
- 💾 **메모리 효율**: 한 번에 2개 프레임만 GPU에 로드

---

## 메모리 관리 전략 비교

### 공통 전략

| 전략 | RIFE | FILM | 설명 |
|------|------|------|------|
| **모델 캐싱** | ❌ | ❌ | 매 실행마다 모델 재로드 |
| **주기적 캐시 클리어** | ✅ | ✅ | 기본 10프레임마다 `soft_empty_cache()` |
| **CPU 출력 누적** | ✅ | ✅ | GPU 대신 CPU 메모리에 결과 저장 |
| **gc.collect()** | ✅ | ✅ | Python 가비지 컬렉터 호출 |
| **최종 정리** | ✅ | ✅ | 완료 후 `soft_empty_cache()` |
| **detach() 사용** | ✅ | ✅ | gradient 추적 제거 |

### 차이점

| 측면 | RIFE | FILM |
|------|------|------|
| **출력 할당** | 사전 할당 (`torch.zeros`) | 동적 리스트 (`list.extend`) |
| **모델 타입** | PyTorch 표준 모델 | TorchScript (JIT) |
| **프레임 처리** | 순차적 타임스텝 | 적응형 재귀적 샘플링 |
| **공통 유틸리티** | `generic_frame_loop()` 사용 | 독립적인 루프 구현 |
| **dtype** | `float16` (메모리 절약) | `float32` (정확도 우선) |

### ComfyUI 통합

둘 다 **ComfyUI의 `model_management` 모듈**을 활용:

```python
from comfy.model_management import soft_empty_cache, get_torch_device

DEVICE = get_torch_device()  # 자동 디바이스 선택

# 주기적 캐시 클리어
soft_empty_cache()  # ComfyUI의 스마트 캐시 관리
```

**`soft_empty_cache()`의 장점**:
- ComfyUI의 다른 노드들과 메모리 협상
- 단순 `torch.cuda.empty_cache()`보다 스마트
- 다른 실행 중인 워크플로우 고려

---

## Best Practices 추출

### 1. 메모리 관리

#### ✅ DO: CPU 기반 출력 누적
```python
# GPU 메모리 압박 방지
output_frames = torch.zeros(..., device="cpu")

# 처리 후 즉시 CPU로 이동
result = model(...).detach().cpu()
```

#### ✅ DO: 주기적 캐시 클리어
```python
if processed_count >= clear_threshold:
    soft_empty_cache()  # ComfyUI 통합
    gc.collect()        # Python GC
    processed_count = 0
```

#### ✅ DO: detach() 사용
```python
# gradient 추적 제거로 메모리 절약
output = model(...).detach()
```

#### ❌ DON'T: GPU에 모든 출력 누적
```python
# 나쁜 예: OOM 위험
output_frames = torch.zeros(..., device="cuda")  # ❌
```

### 2. 디바이스 관리

#### ✅ DO: ComfyUI의 디바이스 선택 활용
```python
from comfy.model_management import get_torch_device
DEVICE = get_torch_device()  # 자동 선택
```

#### ✅ DO: 명시적 전송
```python
frame = frame.to(DEVICE)  # 명확한 의도
result = result.to('cpu') # 명시적 CPU 이동
```

### 3. 모델 로딩

#### 🤔 RIFE/FILM 방식: 캐싱 없음
```python
# 매 실행마다 로드
model = load_model(model_path)
model.eval().to(device)
# ... 사용 후 자동 해제
```

**장점**:
- 간단한 구현
- 메모리 누수 없음
- 다른 모델과 충돌 없음

**단점**:
- 매번 로드 오버헤드 (3-5초)
- 동일 모델 재사용 시 비효율

#### 🎯 대안: 글로벌 캐싱 (TLBVFI 방식)
```python
_MODEL_CACHE = {}

cache_key = f"{model_name}_{gpu_id}"
if cache_key in _MODEL_CACHE:
    model = _MODEL_CACHE[cache_key]
else:
    model = load_model(...)
    _MODEL_CACHE[cache_key] = model
```

**장점**:
- 재사용 시 즉시 실행
- 워크플로우 간 공유

**단점**:
- 메모리 압박 관리 필요
- 명시적 클리어 메커니즘 필요

### 4. 프레임 처리

#### ✅ DO: 사전 할당 (알려진 크기)
```python
# RIFE 스타일: 크기를 알 때
output = torch.zeros(expected_size, device="cpu")
```

#### ✅ DO: 동적 리스트 (가변 크기)
```python
# FILM 스타일: 크기를 모를 때
output_list = []
output_list.extend(frames)
output = torch.cat(output_list)
```

### 5. dtype 관리

#### ✅ DO: 메모리 vs 정확도 트레이드오프
```python
# RIFE: 메모리 우선
output = torch.zeros(..., dtype=torch.float16)

# FILM: 정확도 우선
output = torch.zeros(..., dtype=torch.float32)
```

---

## TLBVFI 구현과의 비교

### 현재 TLBVFI 구현

```python
# nodes/tlbvfi_interpolator.py
_MODEL_CACHE = {}  # 글로벌 캐싱

def interpolate(self, frame_pair, model_name, times_to_interpolate, gpu_id, ...):
    device = torch.device(f"cuda:{gpu_id}")

    # 캐싱 사용
    cache_key = f"{model_name}_{gpu_id}"
    if cache_key in _MODEL_CACHE:
        model = _MODEL_CACHE[cache_key]
    else:
        # 메모리 압박 체크
        if device.type == 'cuda':
            mem_stats = get_memory_stats(device)
            if mem_stats['free'] < 4.0:
                clear_model_cache()

        model = load_tlbvfi_model(model_name, device)
        _MODEL_CACHE[cache_key] = model

    # GPU에서 모든 처리
    current_frames = [frame1, frame2]
    for iteration in range(times_to_interpolate):
        temp_frames = [current_frames[0]]
        for j in range(len(current_frames) - 1):
            with torch.no_grad():
                mid_frame = model.sample(current_frames[j], current_frames[j+1])
            temp_frames.extend([mid_frame, current_frames[j+1]])
        current_frames = temp_frames

    # 후처리: CPU로 이동
    processed_frames = []
    for frame in frames_to_process:
        frame_cpu = frame.squeeze(0).to('cpu', non_blocking=True)
        frame_cpu = (frame_cpu + 1.0) / 2.0
        frame_cpu = frame_cpu.clamp(0, 1)
        frame_cpu = frame_cpu.permute(1, 2, 0)
        processed_frames.append(frame_cpu)

    result = torch.stack(processed_frames, dim=0)

    # 정리
    del current_frames, temp_frames, frame1, frame2
    cleanup_memory(device, force_gc=True)

    return (result,)
```

### 비교 분석

| 측면 | TLBVFI | RIFE/FILM |
|------|--------|-----------|
| **모델 캐싱** | ✅ 글로벌 캐시 | ❌ 매번 로드 |
| **메모리 압박 감지** | ✅ 자동 체크 | ❌ 없음 |
| **처리 위치** | GPU (중간 프레임들) | GPU→CPU 즉시 이동 |
| **출력 누적** | CPU (최종) | CPU (즉시) |
| **캐시 클리어** | 수동 (`force_gc`) | 주기적 (10프레임) |
| **비동기 전송** | ✅ `non_blocking=True` | ❌ 동기 전송 |
| **워크플로우 재사용** | ✅ 모델 유지 | ❌ 매번 로드 |

### 장단점

#### TLBVFI 장점
- ✅ **빠른 재실행**: 모델 캐싱으로 로드 시간 절약
- ✅ **메모리 안전**: 자동 압박 감지 및 클리어
- ✅ **비동기 최적화**: `non_blocking` 전송

#### TLBVFI 단점 (RIFE/FILM 대비)
- ❌ **주기적 정리 없음**: 페어 단위 처리라 덜 필요하지만 고려 가능
- ❌ **ComfyUI 통합 부족**: `soft_empty_cache()` 미사용
- ⚠️ **GPU 중간 누적**: 반복 보간 시 GPU 메모리 압박

---

## 권장사항

### 1. ComfyUI 통합 개선

**현재**:
```python
cleanup_memory(device, force_gc=True)
```

**권장**:
```python
from comfy.model_management import soft_empty_cache
soft_empty_cache()  # ComfyUI의 스마트 캐시 관리
gc.collect()
```

**이유**: ComfyUI의 다른 노드들과 협조적으로 메모리 관리

### 2. 주기적 캐시 클리어 고려 (ChunkProcessor)

`TLBVFI_ChunkProcessor`는 여러 페어를 처리하므로:

```python
# nodes/chunk_processor.py
for pair_idx in tqdm(range(total_pairs)):
    interpolated_frames = self._interpolate_pair(...)
    self._save_chunk_as_video(...)

    # 추가: 주기적 캐시 클리어
    if (pair_idx + 1) % 10 == 0:
        from comfy.model_management import soft_empty_cache
        print(f"Chunk {pair_idx}: Clearing cache...")
        soft_empty_cache()

    cleanup_memory(device, force_gc=True)
```

### 3. 모델 캐싱 전략 유지

TLBVFI의 글로벌 캐싱은 **장점이 많으므로 유지** 권장:

```python
_MODEL_CACHE = {}  # 유지

# 메모리 압박 시 자동 클리어 (현재 구현 유지)
if mem_stats['free'] < 4.0:
    clear_model_cache()
```

**이유**:
- TLBVFI 모델은 3.6GB로 매우 큼
- 재로드 시간이 RIFE/FILM보다 훨씬 김
- 페어 단위 처리로 캐싱 효과 극대화

### 4. 사용자 설정 추가 고려

RIFE/FILM처럼 캐시 클리어 빈도를 사용자가 조절할 수 있도록:

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
                    "max": 100
                }),
            }
        }
```

### 5. 메모리 프로파일링 추가

```python
def process_all_chunks(self, ...):
    for pair_idx in range(total_pairs):
        # 처리 전
        if pair_idx % 50 == 0:  # 50개마다 로깅
            mem_stats = get_memory_stats(device)
            print(f"Pair {pair_idx}: VRAM {mem_stats['used']:.1f}GB / {mem_stats['total']:.1f}GB")

        # 처리...
```

### 6. dtype 최적화 고려

현재 TLBVFI는 `float32` 사용. 메모리 절약을 위해 출력을 `float16`으로:

```python
# 처리는 float32, 저장은 float16
processed_frames = []
for frame in frames_to_process:
    frame_cpu = frame.squeeze(0).to('cpu', non_blocking=True)
    frame_cpu = (frame_cpu + 1.0) / 2.0
    frame_cpu = frame_cpu.clamp(0, 1).to(dtype=torch.float16)  # float16 변환
    processed_frames.append(frame_cpu.permute(1, 2, 0))
```

---

## 결론

### 핵심 발견

1. **RIFE/FILM은 심플함을 선택**: 모델 캐싱 없이 매번 로드
2. **주기적 정리가 핵심**: `soft_empty_cache()` + `gc.collect()`를 10프레임마다
3. **CPU 기반 출력**: GPU 메모리 압박 최소화
4. **ComfyUI 통합 중요**: `soft_empty_cache()`로 다른 노드와 협조

### TLBVFI의 차별화된 장점

- ✅ **모델 캐싱**: 3.6GB 모델의 재로드 오버헤드 제거
- ✅ **메모리 압박 감지**: 자동 캐시 클리어
- ✅ **페어 단위 처리**: 메모리 효율적 설계
- ✅ **비동기 전송**: 성능 최적화

### 개선 제안 우선순위

1. **높음**: ComfyUI의 `soft_empty_cache()` 통합
2. **중간**: ChunkProcessor에 주기적 캐시 클리어 추가
3. **낮음**: 사용자 설정 가능한 캐시 클리어 빈도
4. **선택**: dtype 최적화 (float16 출력)

---

## 참고자료

- [ComfyUI-Frame-Interpolation Repository](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation)
- [RIFE Paper (ECCV 2022)](https://arxiv.org/abs/2011.06294)
- [FILM Paper (ECCV 2022)](https://arxiv.org/abs/2202.04901)
- [ComfyUI Model Management](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_management.py)

---

**분석 완료**: 2025-10-15
**다음 단계**: 위 권장사항 중 높은 우선순위 항목부터 구현 검토
