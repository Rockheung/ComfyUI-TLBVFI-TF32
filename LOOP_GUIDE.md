# ComfyUI 루프를 사용한 워크플로우 가이드

ComfyUI-TLBVFI-TF32는 두 가지 워크플로우 패턴을 지원합니다:

## 방법 1: 올인원 노드 (권장) ⭐

**TLBVFI_ChunkProcessor** 노드를 사용하면 루프 없이 자동으로 모든 프레임 페어를 처리합니다.

### 워크플로우:
```
VHS LoadVideo → TLBVFI_ChunkProcessor → TLBVFI_VideoConcatenator
```

### 장점:
- ✅ 추가 커스텀 노드 불필요
- ✅ 자동 순회 (수동 관리 없음)
- ✅ 모델 자동 재사용
- ✅ 진행 상황 표시 (tqdm)
- ✅ 가장 간단한 방법

### 사용법:
1. `TLBVFI_ChunkProcessor` 노드 추가
2. VHS LoadVideo의 IMAGE 연결
3. 파라미터 설정 (model, times_to_interpolate, fps, codec, crf)
4. 실행 → 모든 페어 자동 처리
5. session_id를 VideoConcatenator에 연결

---

## 방법 2: ComfyUI 루프 노드 사용

커스텀 노드를 설치하여 ComfyUI의 루프 기능을 활용할 수 있습니다.

### A. ComfyUI-Easy-Use 사용 (전통적 For Loop)

#### 설치:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yolain/ComfyUI-Easy-Use.git
cd ComfyUI-Easy-Use
pip install -r requirements.txt
```

#### 워크플로우:
```
VHS LoadVideo → TLBVFI_FramePairSlicer
                     ↓
              easy forLoopStart (iterations 계산 필요)
                     ↓
              TLBVFI_Interpolator
                     ↓
              TLBVFI_ChunkVideoSaver
                     ↓
              easy forLoopEnd
                     ↓
              TLBVFI_VideoConcatenator
```

#### 설정 방법:

1. **VHS LoadVideo**로 비디오 로드
2. **TLBVFI_FramePairSlicer** 추가:
   - `images`: VHS LoadVideo 출력 연결
   - `pair_index`: `easy forLoopStart`의 `loop_index` 출력 연결
3. **easy forLoopStart** 추가:
   - `iterations`: FramePairSlicer의 `total_pairs` 출력 연결
   - `loop_index` 출력을 FramePairSlicer의 `pair_index`에 연결
4. **TLBVFI_Interpolator** 추가:
   - `frame_pair`: FramePairSlicer 출력
   - `shared_model`: 자기 자신의 `shared_model` 출력을 입력으로 연결 (루프)
   - `is_last_pair`: FramePairSlicer의 `is_last_pair` 연결
5. **TLBVFI_ChunkVideoSaver** 추가:
   - `interpolated_frames`: Interpolator 출력
   - `session_id`: 첫 실행시 생성되어 유지됨
6. **easy forLoopEnd** 추가:
   - Interpolator 출력 연결 (수집만 함, 사용하진 않음)
7. **TLBVFI_VideoConcatenator** 추가:
   - `session_id`: ChunkVideoSaver의 session_id 연결

#### 장점:
- ✅ 프로그래밍의 for 루프와 동일한 패턴
- ✅ 명확한 반복 구조

#### 단점:
- ❌ 추가 커스텀 노드 설치 필요
- ❌ total_pairs 계산을 위해 FramePairSlicer 미리 실행 필요
- ❌ 복잡한 연결 구조

---

### B. ComfyUI-Impact-Pack 사용 (조건부 While Loop)

#### 설치:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git
cd ComfyUI-Impact-Pack
pip install -r requirements.txt
```

#### 워크플로우:
```
VHS LoadVideo → TLBVFI_FramePairSlicer
                     ↓
              ImpactInt (카운터)
                     ↓
              TLBVFI_Interpolator
                     ↓
              TLBVFI_ChunkVideoSaver
                     ↓
              ImpactCompare (pair_index < total_pairs)
                     ↓
              ImpactConditionalStopIteration
```

#### 설정 방법:

1. **ComfyUI Auto Queue 활성화** (필수!):
   - UI 우상단 "Extra options" → "Auto Queue" 체크
2. **VHS LoadVideo**로 비디오 로드
3. **ImpactInt** 추가:
   - `value`: 0 (시작시 **반드시 수동으로 0으로 리셋**)
   - `mode`: "increment" 또는 직접 카운터 관리
4. **TLBVFI_FramePairSlicer** 추가:
   - `images`: VHS LoadVideo 출력
   - `pair_index`: ImpactInt의 `int` 출력 연결
5. **TLBVFI_Interpolator** 추가:
   - `frame_pair`: FramePairSlicer 출력
   - `shared_model`: 자기 자신의 출력을 입력으로 연결
   - `is_last_pair`: FramePairSlicer의 `is_last_pair` 연결
6. **TLBVFI_ChunkVideoSaver** 추가:
   - `interpolated_frames`: Interpolator 출력
7. **ImpactCompare** 추가:
   - `a`: FramePairSlicer의 `pair_index` 출력
   - `b`: FramePairSlicer의 `total_pairs` 출력
   - `comparison`: "a < b"
8. **ImpactConditionalStopIteration** 추가:
   - `condition`: ImpactCompare의 `boolean` 출력
   - 조건이 False가 되면 루프 종료
9. **TLBVFI_VideoConcatenator** 추가:
   - `session_id`: ChunkVideoSaver의 session_id

#### 장점:
- ✅ 조건부 종료 가능 (while 루프)
- ✅ 유연한 비교 로직
- ✅ 복잡한 워크플로우에 적합

#### 단점:
- ❌ Auto Queue 필수
- ❌ **카운터 수동 리셋 필요** (캐시 문제)
- ❌ 설정이 복잡함
- ❌ 추가 커스텀 노드 설치 필요

---

### C. ComfyUI-Loop (이미지 Save/Load 패턴)

#### 설치:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Hullabalo/ComfyUI-Loop.git
```

#### 워크플로우:
```
Load Image (LOOP) [/tmp/loop_input.png]
    ↓
[카운터 증가 로직]
    ↓
TLBVFI_FramePairSlicer (pair_index from counter)
    ↓
TLBVFI_Interpolator
    ↓
TLBVFI_ChunkVideoSaver
    ↓
Save Image (LOOP) [/tmp/loop_input.png]
```

#### 특징:
- ✅ 간단한 피드백 루프
- ✅ 150회 이상 반복해도 품질 저하 없음
- ❌ 카운터 관리가 복잡함 (이미지에 인코딩 필요)
- ❌ 이 워크플로우에는 부적합

---

## 방법 3: comfyui-job-iterator (시퀀스 처리)

### 설치:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/O-oshir/comfy-loop-utilities.git
```

### 워크플로우:
```
VHS LoadVideo → [프레임 수 계산]
                     ↓
              MakeJob (sequence=[0,1,2,...,N-1])
                     ↓
              JobIterator
                     ↓
              TLBVFI_FramePairSlicer (pair_index from iterator)
                     ↓
              TLBVFI_Interpolator
                     ↓
              TLBVFI_ChunkVideoSaver
                     ↓
              TLBVFI_VideoConcatenator
```

### 장점:
- ✅ 단일 실행으로 모든 값 처리
- ✅ 재큐잉 불필요
- ✅ 시퀀스 명확하게 정의

### 단점:
- ❌ 시퀀스를 미리 생성해야 함
- ❌ 추가 커스텀 노드 설치 필요

---

## 비교표

| 방법 | 추가 노드 필요 | 난이도 | Auto Queue | 수동 관리 | 추천도 |
|------|--------------|--------|------------|----------|--------|
| **TLBVFI_ChunkProcessor** | ❌ | ⭐ 쉬움 | ❌ | ❌ | ⭐⭐⭐⭐⭐ |
| ComfyUI-Easy-Use | ✅ | ⭐⭐ 보통 | ❌ | total_pairs | ⭐⭐⭐⭐ |
| ComfyUI-Impact-Pack | ✅ | ⭐⭐⭐ 어려움 | ✅ 필수 | 카운터 리셋 | ⭐⭐⭐ |
| comfyui-job-iterator | ✅ | ⭐⭐ 보통 | ❌ | 시퀀스 생성 | ⭐⭐⭐ |
| ComfyUI-Loop | ✅ | ⭐⭐⭐⭐ 매우 어려움 | ✅ | 카운터 인코딩 | ⭐ |

---

## 권장사항

### 1. 대부분의 경우:
**TLBVFI_ChunkProcessor 노드 사용** - 가장 간단하고 효율적

### 2. 커스터마이징이 필요한 경우:
**ComfyUI-Easy-Use** - 전통적인 for 루프 패턴

### 3. 복잡한 조건부 로직이 필요한 경우:
**ComfyUI-Impact-Pack** - while 루프 같은 조건부 제어

---

## 주요 제약사항

### ComfyUI 루프의 일반적 제한:

1. **RecursionError**: 4일 이상 연속 실행시 Python 재귀 깊이 초과 가능
2. **캐시 문제**: 루프 카운터가 이전 워크플로우 상태 유지 → ComfyUI 재시작 필요
3. **Auto Queue 요구**: Impact-Pack, ComfyUI-Loop 등은 Auto Queue 필수
4. **직접 재귀 불가**: 표준 ComfyUI는 output → input 직접 연결 불가
5. **메모리 주의**: 루프 내 배치 처리시 메모리 곱셈 효과

### FramePairSlicer 업데이트:

- `pair_index`가 이제 **optional** 파라미터
- 루프 카운터 연결 가능 (forceInput: False)
- 기본값 0으로 위젯으로도 사용 가능

---

## 예시 워크플로우 (ComfyUI-Easy-Use)

```json
{
  "1": {
    "class_type": "VHS_LoadVideo",
    "inputs": {
      "video": "input_video.mp4"
    }
  },
  "2": {
    "class_type": "TLBVFI_FramePairSlicer",
    "inputs": {
      "images": ["1", 0],
      "pair_index": ["5", 0]  // From forLoopStart
    }
  },
  "3": {
    "class_type": "TLBVFI_Interpolator",
    "inputs": {
      "frame_pair": ["2", 0],
      "model_name": "vimeo_unet.pth",
      "times_to_interpolate": 1,
      "gpu_id": 0,
      "shared_model": ["3", 1],  // Self-loop
      "is_last_pair": ["2", 3]
    }
  },
  "4": {
    "class_type": "TLBVFI_ChunkVideoSaver",
    "inputs": {
      "interpolated_frames": ["3", 0],
      "session_id": "my_session",
      "fps": 30.0,
      "codec": "h264_nvenc",
      "crf": 18
    }
  },
  "5": {
    "class_type": "easy forLoopStart",
    "inputs": {
      "iterations": ["2", 2]  // total_pairs
    }
  },
  "6": {
    "class_type": "easy forLoopEnd",
    "inputs": {
      "flow": ["3", 0]  // Just for collection
    }
  },
  "7": {
    "class_type": "TLBVFI_VideoConcatenator",
    "inputs": {
      "session_id": ["4", 0]
    }
  }
}
```

---

## 문제 해결

### Q: "Auto Queue를 활성화해야 한다"는 메시지가 나옴
**A**: UI 우상단 "Extra options" → "Auto Queue" 체크

### Q: 루프가 무한 반복됨
**A**: Impact-Pack 사용시 카운터를 0으로 수동 리셋

### Q: FramePairSlicer에 pair_index를 연결할 수 없음
**A**: 최신 버전(0.3.0+)으로 업데이트 - pair_index가 optional로 변경됨

### Q: RecursionError 발생
**A**: ComfyUI 재시작 필요 (장시간 실행시 발생)

### Q: 루프가 첫 반복만 실행되고 멈춤
**A**: Auto Queue 활성화 또는 TLBVFI_ChunkProcessor 사용

---

## 결론

- **간단한 사용**: `TLBVFI_ChunkProcessor` 사용 ⭐
- **커스터마이징**: `ComfyUI-Easy-Use` 설치 후 for 루프 사용
- **고급 제어**: `ComfyUI-Impact-Pack`으로 조건부 루프

대부분의 경우 **TLBVFI_ChunkProcessor**를 사용하는 것이 가장 효율적이고 간단합니다!
