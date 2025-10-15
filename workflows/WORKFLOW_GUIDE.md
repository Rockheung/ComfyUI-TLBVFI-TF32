# TLBVFI V2 워크플로우 구성 가이드

## ⚠️ 중요: Dependency Cycle 오류 해결

제공된 JSON 워크플로우가 ComfyUI의 실제 구조와 맞지 않아 "Dependency cycle detected" 오류가 발생할 수 있습니다.

**해결 방법**: ComfyUI 내에서 직접 노드를 배치하여 워크플로우를 만드세요.

---

## 🎯 기본 V2 워크플로우 (수동 구성)

### 필요한 노드:
1. **LoadImage** (×2) - 이전 프레임, 다음 프레임
2. **TLBVFI_Interpolator_V2** - 보간 처리
3. **PreviewImage** - 결과 확인
4. **SaveImage** (선택) - 결과 저장

### 연결 방법:

```
LoadImage (prev) ──┐
                   ├──> TLBVFI_Interpolator_V2 ──> PreviewImage
LoadImage (next) ──┘                            └──> SaveImage
```

### 단계별 구성:

1. **ComfyUI 시작** → 빈 캔버스

2. **LoadImage 노드 추가** (2개)
   - 우클릭 → `Add Node` → `image` → `LoadImage`
   - 또는 더블클릭 후 "LoadImage" 검색
   - 하나는 이전 프레임용, 하나는 다음 프레임용

3. **TLBVFI_Interpolator_V2 추가**
   - 우클릭 → `Add Node` → `frame_interpolation` → `TLBVFI-TF32` → `TLBVFI Interpolator V2 [Production]`
   - 또는 검색: "TLBVFI V2"

4. **연결하기**:
   - LoadImage(1)의 `IMAGE` 출력 → V2의 `prev_frame` 입력
   - LoadImage(2)의 `IMAGE` 출력 → V2의 `next_frame` 입력

5. **PreviewImage 추가**:
   - V2의 `interpolated_frames` 출력 → PreviewImage의 `images` 입력

6. **V2 노드 설정**:
   ```
   model_name: vimeo_unet.pth
   times_to_interpolate: 0 (단일 프레임)
   enable_tf32: ✓ (체크)
   sample_steps: 10
   flow_scale: 0.5
   cpu_offload: ✓ (체크)
   gpu_id: 0
   ```

7. **Queue Prompt** 실행

---

## 🎬 비디오 처리 워크플로우

### VHS LoadVideo 사용

```
VHS_LoadVideo ──> TLBVFI_FramePairSlicer ──┐
                                            ├──> TLBVFI_Interpolator_V2
                                            │    (prev: pair[0], next: pair[1])
                                            │         │
                                            │         v
                                            │    PreviewImage
                                            │         │
                                            │         v
                                            └────> (다음 pair로 반복)
```

### 단계별:

1. **VHS_LoadVideo 추가**
   - `Add Node` → `Video Helper Suite` → `VHS_LoadVideo`
   - 비디오 파일 선택

2. **TLBVFI_FramePairSlicer 추가**
   - `Add Node` → `frame_interpolation` → `TLBVFI-TF32` → `TLBVFI Frame Pair Slicer`
   - VHS_LoadVideo의 `IMAGE` → FramePairSlicer의 `images`
   - `pair_index`: 0으로 시작

3. **프레임 분리하기**

   FramePairSlicer는 2개의 프레임을 하나의 배치로 출력합니다 `(2, H, W, C)`.

   **방법 1: TLBVFI_FrameFromBatch 사용** (추천)
   ```
   FramePairSlicer ──> FrameFromBatch [index: 0] ──> V2 prev_frame
                   └──> FrameFromBatch [index: 1] ──> V2 next_frame
   ```

   **방법 2: LatentFromBatch 사용**
   ```
   FramePairSlicer ──> LatentFromBatch [batch_index: 0] ──> V2 prev_frame
                   └──> LatentFromBatch [batch_index: 1] ──> V2 next_frame
   ```

   **방법 3: ImageBatch 노드 사용**
   - ComfyUI 기본 노드 중 배치 분리 노드 활용

4. **TLBVFI_Interpolator_V2 연결**
   - prev_frame: 첫 번째 프레임
   - next_frame: 두 번째 프레임

5. **반복 처리**:
   - 첫 실행: `pair_index = 0`
   - 두 번째: `pair_index = 1`
   - 세 번째: `pair_index = 2`
   - ... (total_pairs까지)

---

## 🔧 실전 예제: 2개 프레임 보간

### 가장 간단한 테스트:

1. **이미지 2장 준비**:
   - `frame_001.png` (첫 번째 프레임)
   - `frame_002.png` (두 번째 프레임)
   - ComfyUI `input` 폴더에 저장

2. **노드 배치**:
   ```
   [LoadImage: frame_001.png] ─┐
                                ├─> [TLBVFI_Interpolator_V2] ─> [PreviewImage]
   [LoadImage: frame_002.png] ─┘
   ```

3. **V2 설정**:
   - `times_to_interpolate`: 1 (3 프레임 생성)
   - 나머지 기본값 사용

4. **실행 → 3개 프레임 출력**:
   - 프레임 1 (원본)
   - 프레임 1.5 (보간)
   - 프레임 2 (원본)

---

## 💡 자주 묻는 질문

### Q1: "Dependency cycle detected" 오류가 계속 나요

**A**: JSON 워크플로우 대신 UI에서 직접 만드세요.
- JSON 파일은 참고용입니다
- ComfyUI의 노드 배치가 더 안정적입니다
- 위 가이드를 따라 수동으로 구성하세요

### Q2: FramePairSlicer 출력을 어떻게 분리하나요?

**A**: 표준 ComfyUI 노드를 사용하세요:

**옵션 1** - ImageBatch 관련 노드:
```
FramePairSlicer ──> [Custom 배치 분리 노드]
```

**옵션 2** - 직접 수정:
FramePairSlicer 출력이 `(2, H, W, C)`이므로, ComfyUI의 배치 처리 노드로 인덱싱

**옵션 3** - 단순화된 접근:
두 개의 LoadImage로 프레임을 개별로 로드 (권장)

### Q3: 비디오 전체를 한 번에 처리할 수 없나요?

**A**: V2는 메모리 안전성을 위해 프레임 쌍 단위로 처리합니다.

**자동화 방법**:
1. ComfyUI API 사용 (Python 스크립트)
2. ComfyUI-Manager의 Queue 기능
3. 외부 스크립트로 pair_index 증가하며 반복
4. 또는 **TLBVFI_BatchInterpolator_V2** 노드로 한 번에 전체 배치를 처리

**예제 스크립트** (`process_video.py`):
```python
import requests
import json
import time

api_url = "http://127.0.0.1:8188/prompt"
workflow_file = "my_workflow.json"

# 워크플로우 로드
with open(workflow_file) as f:
    workflow = json.load(f)

total_pairs = 100  # 비디오의 프레임 수 - 1

for pair_idx in range(total_pairs):
    # pair_index 업데이트 (노드 ID는 실제 워크플로우에 맞게 조정)
    for node in workflow["prompt"].values():
        if node.get("class_type") == "TLBVFI_FramePairSlicer":
            node["inputs"]["pair_index"] = pair_idx
            break

    # Queue에 추가
    response = requests.post(api_url, json={"prompt": workflow})
    print(f"Queued pair {pair_idx + 1}/{total_pairs}")

    # 완료 대기 (선택사항)
    time.sleep(10)  # 또는 status API로 확인
```

### Q4: 메모리 부족 오류가 나요

**A**: V2 설정 확인:
- `cpu_offload`: 반드시 체크
- `times_to_interpolate`: 낮추기 (4→2→1→0)
- ComfyUI 재시작으로 캐시 클리어

### Q5: 속도가 너무 느려요

**A**:
- `sample_steps`: 10으로 설정 (기본값)
- `flow_scale`: 0.5로 설정
- `enable_tf32`: 체크 (RTX 30/40만)
- GPU 드라이버 최신 버전 확인

---

## 📊 권장 설정표

### 해상도별:

| 해상도 | times_to_interpolate | cpu_offload | 예상 VRAM (TF32) |
|--------|---------------------|-------------|------------------|
| 720p | 3 | ✓ | ~3GB |
| 1080p | 2 | ✓ | ~3.5GB |
| 4K | 1 | ✓ | ~4.2GB |
| 8K | 0 | ✓ | ~8GB |

### 용도별:

| 용도 | sample_steps | flow_scale | 예상 시간/쌍 |
|------|-------------|-----------|--------------|
| 빠른 프리뷰 | 10 | 0.5 | ~8s (RTX 3090) |
| 프로덕션 기본 | 20 | 0.5 | ~16s |
| 최종 납품 | 50 | 1.0 | ~40s |

---

## 🛠️ 트러블슈팅

### 노드가 안 보여요
- ComfyUI 재시작
- `custom_nodes/ComfyUI-TLBVFI-TF32` 폴더 확인
- `__init__.py`에서 노드 등록 확인

### 모델을 찾을 수 없어요
- `ComfyUI/models/interpolation/vimeo_unet.pth` 확인
- 파일 크기: ~3.6GB
- 다운로드: https://huggingface.co/ucfzl/TLBVFI

### 결과가 이상해요
- 입력 프레임 순서 확인 (prev → next)
- 프레임 형식 확인 (RGB, 0-1 범위)
- `times_to_interpolate` 설정 확인

---

## 📚 추가 리소스

- **메인 README**: `../README.md`
- **V2 설계 문서**: `../docs/Production-Improvement-Plan.md`
- **원본 논문 분석**: `../docs/TLBVFI-Original-Implementation-Analysis.md`
- **GitHub Issues**: https://github.com/Rockheung/ComfyUI-TLBVFI-TF32/issues

---

## 💾 워크플로우 저장

ComfyUI에서 만든 워크플로우를 저장하려면:
1. 메뉴에서 `Save` 클릭
2. JSON 파일로 저장
3. 나중에 `Load`로 불러오기

**팁**: 잘 동작하는 워크플로우를 템플릿으로 저장해두세요!
