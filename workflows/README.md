# TLBVFI V2 Workflow Examples

이 디렉토리에는 TLBVFI Interpolator V2를 활용하는 다양한 워크플로우 예제가 포함되어 있습니다.

## 워크플로우 목록

### 1. `01_basic_v2_interpolation.json` - 기본 보간 워크플로우

**용도**: V2 노드를 처음 사용하는 경우, 단일 프레임 쌍 보간

**구조**:
```
VHS_LoadVideo → FramePairSlicer → TLBVFI_Interpolator_V2 → VHS_VideoCombine
                                                          └→ PreviewImage
```

**주요 설정**:
- `times_to_interpolate`: 1 (3 프레임 출력)
- `enable_tf32`: True (RTX 30/40)
- `sample_steps`: 10 (빠른 처리)
- `cpu_offload`: True (메모리 안전)

**사용 방법**:
1. VHS_LoadVideo에서 비디오 로드
2. FramePairSlicer의 `pair_index`를 0부터 시작
3. 각 실행마다 `pair_index`를 1씩 증가
4. 결과를 프리뷰하거나 비디오로 저장

**메모리**: ~4.2GB VRAM (4K, TF32 + cpu_offload)

---

### 2. `02_batch_loop_interpolation.json` - 배치 루프 워크플로우

**용도**: 전체 비디오의 모든 프레임 쌍을 자동으로 처리

**구조**:
```
VHS_LoadVideo → FramePairSlicer → TLBVFI_Interpolator_V2 → ImageBatchToList
                                                          → ImageListToImageBatch
                                                          → VHS_VideoCombine
                                                          → PreviewImage
```

**주요 기능**:
- **프레임 축적**: ImageListToImageBatch가 이전 결과를 누적
- **진행 상황 모니터링**: StringFormat + ShowText로 실시간 진행률 표시
- **오디오 보존**: 원본 비디오의 오디오 트랙 유지
- **메모리 안전**: 각 반복마다 ~4.2GB만 사용

**사용 방법**:

#### 수동 방식:
1. `pair_index = 0`으로 시작
2. Queue Prompt 실행
3. 완료 후 `pair_index`를 1 증가
4. total_pairs에 도달할 때까지 반복

#### 자동화 방식 (ComfyUI API):
```python
import requests
import json

# ComfyUI API 엔드포인트
api_url = "http://127.0.0.1:8188/prompt"

# 워크플로우 로드
with open("02_batch_loop_interpolation.json") as f:
    workflow = json.load(f)

# 전체 프레임 쌍 수 (비디오 프레임 수 - 1)
total_pairs = 100  # 예: 101 프레임 비디오

for pair_index in range(total_pairs):
    # pair_index 업데이트
    workflow["nodes"][1]["widgets_values"]["pair_index"] = pair_index

    # Queue에 추가
    payload = {"prompt": workflow}
    response = requests.post(api_url, json=payload)

    print(f"Queued pair {pair_index}/{total_pairs}")

    # 선택사항: 완료 대기 로직 추가 가능
```

**예상 시간** (RTX 3090, 4K, times_to_interpolate=2):
- 프레임 쌍당: ~10초
- 100 쌍: ~16분
- 1000 쌍: ~2.7시간

---

### 3. `03_quality_comparison.json` - 품질 비교 워크플로우

**용도**: 다양한 품질 설정을 동시에 비교

**구조**:
```
VHS_LoadVideo → FramePairSlicer → [Fast V2] → PreviewImage
                                → [Balanced V2] → PreviewImage
                                → [Quality V2] → PreviewImage
                                           ↓
                                    ImageCompare → SaveImage
```

**비교 설정**:

| 모드 | sample_steps | flow_scale | 속도 (4K) | 품질 |
|------|-------------|-----------|----------|------|
| **Fast** | 10 | 0.5 | ~8s | 양호 |
| **Balanced** | 20 | 0.5 | ~16s | 우수 |
| **Quality** | 50 | 1.0 | ~40s | 최고 |

**사용 시나리오**:
1. **품질 테스트**: 어떤 설정이 당신의 콘텐츠에 적합한지 확인
2. **벤치마킹**: 하드웨어 성능 측정
3. **A/B 테스트**: 클라이언트에게 결과 제시

**주의사항**:
- 3개 모델 인스턴스 동시 실행 → VRAM ~12GB 필요
- 8GB GPU의 경우: 한 번에 하나씩 실행 (나머지 비활성화)

---

## 공통 팁 및 베스트 프랙티스

### 메모리 관리

**8GB GPU의 경우**:
```json
{
  "enable_tf32": true,
  "cpu_offload": true,
  "times_to_interpolate": 2
}
```

**12GB+ GPU의 경우**:
```json
{
  "enable_tf32": true,
  "cpu_offload": false,
  "times_to_interpolate": 3
}
```

**24GB+ GPU의 경우**:
```json
{
  "enable_tf32": true,
  "cpu_offload": false,
  "times_to_interpolate": 4
}
```

### 품질 vs 속도 선택 가이드

**빠른 프리뷰** (초안, 테스트):
- `sample_steps`: 10
- `flow_scale`: 0.5
- 속도: 1x (기준)

**프로덕션 기본** (대부분의 경우):
- `sample_steps`: 20
- `flow_scale`: 0.5
- 속도: 2x
- 품질: 눈에 띄는 개선

**최고 품질** (최종 납품, 아카이브):
- `sample_steps`: 50
- `flow_scale`: 1.0
- 속도: 5x
- 품질: 최고 수준

### 해상도별 권장 설정

| 해상도 | times_to_interpolate | sample_steps | 예상 VRAM (TF32) |
|--------|---------------------|--------------|------------------|
| 720p (HD) | 4 (16x) | 20 | ~3GB |
| 1080p (FHD) | 3 (8x) | 20 | ~3.5GB |
| 1440p (2K) | 2 (4x) | 20 | ~3.8GB |
| 2160p (4K) | 2 (4x) | 10 | ~4.2GB |
| 4320p (8K) | 1 (2x) | 10 | ~8GB |

### 트러블슈팅

#### OOM 에러 발생 시:
1. `cpu_offload`를 `true`로 설정
2. `times_to_interpolate`를 낮춤 (4→3→2→1→0)
3. 다른 ComfyUI 노드나 프로그램 종료
4. ComfyUI 재시작 (캐시 클리어)

#### 속도가 느릴 때:
1. `enable_tf32`가 `true`인지 확인 (RTX 30/40만 해당)
2. `sample_steps`를 낮춤 (50→20→10)
3. `flow_scale`을 0.5로 설정
4. `cpu_offload`를 `false`로 변경 (충분한 VRAM이 있다면)
5. GPU 드라이버 최신 버전 확인

#### 품질이 기대 이하일 때:
1. `sample_steps`를 높임 (10→20→50)
2. `flow_scale`을 1.0으로 설정
3. 입력 비디오 품질 확인 (압축, 노이즈 등)

---

## 추가 리소스

### 참고 문서:
- `../docs/Production-Improvement-Plan.md` - V2 설계 문서
- `../docs/TLBVFI-Original-Implementation-Analysis.md` - 원본 논문 분석
- `../docs/ComfyUI-Frame-Interpolation-Analysis.md` - RIFE/FILM 패턴 분석
- `../README.md` - 메인 문서

### 원본 논문:
- **GitHub**: https://github.com/ZonglinL/TLBVFI
- **Project Page**: https://zonglinl.github.io/tlbvfi_page/
- **arXiv**: https://arxiv.org/abs/2507.04984

### ComfyUI 리소스:
- **VHS Video Helper Suite**: https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
- **ComfyUI Manager**: https://github.com/ltdrdata/ComfyUI-Manager
- **ComfyUI API**: https://github.com/comfyanonymous/ComfyUI/wiki/API

---

## 피드백 및 기여

워크플로우 개선 제안이나 새로운 사용 사례가 있다면:
- GitHub Issues: https://github.com/Rockheung/ComfyUI-TLBVFI-TF32/issues
- Pull Requests: 환영합니다!

커뮤니티 기여 워크플로우는 `community/` 폴더에 추가됩니다.
