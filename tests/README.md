# Testing Guide

ComfyUI-TLBVFI-TF32 테스트 시스템 가이드

## 📋 목차

- [빠른 시작](#빠른-시작)
- [테스트 종류](#테스트-종류)
- [로컬 테스트 실행](#로컬-테스트-실행)
- [GPU 테스트](#gpu-테스트)
- [CI/CD](#cicd)
- [문제 해결](#문제-해결)

## 🚀 빠른 시작

### 기본 설정

#### Option A: uv 사용 (권장 - 빠르고 현대적)

```bash
# 1. 레포지토리 클론
git clone https://github.com/Rockheung/ComfyUI-TLBVFI-TF32.git
cd ComfyUI-TLBVFI-TF32

# 2. uv 설치 (미설치 시)
# https://docs.astral.sh/uv/getting-started/installation/

# 3. 의존성 설치
uv sync --all-extras

# 4. 테스트 실행
uv run pytest tests/ -v
```

#### Option B: pip 사용 (전통적 방식)

```bash
# 1. 레포지토리 클론
git clone https://github.com/Rockheung/ComfyUI-TLBVFI-TF32.git
cd ComfyUI-TLBVFI-TF32

# 2. 의존성 설치
pip install -r requirements.txt
# 또는
pip install ".[dev]"

# 3. 테스트 실행
pytest tests/ -v
```

## 📊 테스트 종류

### 1. 단위 테스트 (Unit Tests)
- **위치**: `tests/test_frame_pair_slicer.py`, `tests/test_node_registration.py`
- **실행 시간**: ~2-3초
- **요구사항**: None (모든 의존성 mock 처리됨)
- **테스트 수**: 48개

```bash
# 단위 테스트만 실행
pytest tests/ -m unit -v
```

### 2. 통합 테스트 (Integration Tests)
- **위치**: `tests/test_integration.py`
- **실행 시간**: ~1-2초
- **요구사항**: 테스트 영상 파일 (examples/vfi_test_*.mp4)
- **테스트 수**: 11개

```bash
# 통합 테스트만 실행
pytest tests/test_integration.py -v
```

### 3. GPU 테스트
- **마커**: `@pytest.mark.requires_gpu`
- **요구사항**: CUDA GPU
- **테스트 수**: 1개 (추가 가능)

```bash
# GPU 테스트 포함 실행
pytest tests/ -v -m "not slow"
```

## 💻 로컬 테스트 실행

### 전체 테스트 실행

```bash
# 모든 테스트 실행 (권장)
pytest tests/ -v

# 간단한 출력
pytest tests/

# 실패 시 상세 정보
pytest tests/ -v --tb=long

# 특정 테스트만 실행
pytest tests/test_frame_pair_slicer.py::TestFramePairSlicerBasics -v
```

### 마커별 실행

```bash
# 단위 테스트만
pytest tests/ -m unit -v

# 통합 테스트만
pytest tests/ -m integration -v

# GPU 테스트만
pytest tests/ -m requires_gpu -v

# 느린 테스트 제외
pytest tests/ -m "not slow" -v
```

### 커버리지 측정

```bash
# 커버리지와 함께 실행
pytest tests/ --cov=nodes --cov=utils --cov-report=html --cov-report=term

# 커버리지 리포트 보기 (HTML)
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## 🎮 GPU 테스트

### RTX 4090 머신에서 전체 테스트

#### 1. 환경 준비 (Windows PowerShell)

```powershell
# CUDA 버전 확인
nvidia-smi

# PyTorch CUDA 지원 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0)}')"
```

#### 2. 전체 테스트 실행

**Option A: uv 사용 (권장)**

```powershell
# uv 설치 (미설치 시)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 의존성 설치 (CUDA 포함)
cd ComfyUI-TLBVFI-TF32
uv sync --extra cuda --extra dev

# 모든 테스트 실행 (GPU 포함)
uv run pytest tests/ -v --tb=short

# GPU 테스트만 실행
uv run pytest tests/ -v -m requires_gpu

# 느린 테스트 포함
uv run pytest tests/ -v --run-slow
```

**Option B: pip 사용 (전통적)**

```powershell
# pip로 의존성 설치
pip install pytest pytest-cov opencv-python

# 모든 테스트 실행 (GPU 포함)
python -m pytest tests/ -v --tb=short

# GPU 테스트만 실행
python -m pytest tests/ -v -m requires_gpu

# 느린 테스트 포함
python -m pytest tests/ -v --run-slow
```

#### 3. 예상 결과

```
======================== test session starts =========================
platform linux -- Python 3.11.x, pytest-8.x.x, pluggy-1.x.x
collected 59 items

tests/test_frame_pair_slicer.py::...::test_works_with_gpu_tensors PASSED  ✅
tests/test_integration.py::... (11 passed)
tests/test_node_registration.py::... (25 passed)

================= 59 passed in 3.5s =================
```

### GPU 테스트 추가 예제

현재 GPU 테스트가 1개뿐이므로, 더 추가하려면:

```python
# tests/test_frame_pair_slicer.py에 추가
@pytest.mark.requires_gpu
def test_large_batch_on_gpu(self, sample_frames_100, skip_if_no_gpu):
    """Test processing large batch on GPU."""
    gpu_frames = sample_frames_100.cuda()
    slicer = FramePairSlicer()

    frame_pair, _, _, _ = slicer.slice_pair(gpu_frames, pair_index=50)

    assert frame_pair.is_cuda
    assert frame_pair.device.type == 'cuda'
```

## 🔄 CI/CD

### GitHub Actions 자동 실행

**트리거:**
- Push to `main` or `develop`
- Pull Request to `main` or `develop`

**실행 내용:**
```yaml
- Python 3.10, 3.11, 3.12에서 테스트
- 단위 테스트 + 통합 테스트 (GPU 제외)
- 코드 커버리지 측정
- Linting (flake8, black, isort)
```

**결과 확인:**
https://github.com/Rockheung/ComfyUI-TLBVFI-TF32/actions

### 로컬에서 CI와 동일한 환경 테스트

```bash
# Python 버전별 테스트 (pyenv 사용)
pyenv local 3.10.x
pytest tests/ -v
pyenv local 3.11.x
pytest tests/ -v
pyenv local 3.12.x
pytest tests/ -v

# 또는 tox 사용
pip install tox
tox
```

## 🛠️ 문제 해결

### 테스트가 실패하는 경우

#### Windows에서 'pytest' 명령을 찾을 수 없는 경우

**증상:**
```powershell
pytest : 'pytest' 용어가 cmdlet, 함수, 스크립트 파일 또는 실행할 수 있는 프로그램 이름으로 인식되지 않습니다.
```

**해결 방법 (uv 사용 권장):**

```powershell
# 1. uv 설치 (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 의존성 설치
cd ComfyUI-TLBVFI-TF32
uv sync --extra cuda --extra dev

# 3. 테스트 실행
uv run pytest tests/ -v -m "not slow"
```

**대안 (pip 사용):**

```powershell
# 1. pip로 의존성 설치
pip install pytest pytest-cov opencv-python

# 2. Python 모듈로 직접 실행
python -m pytest tests/ -v -m "not slow"
```

#### Linux/macOS에서 ImportError: No module named 'pytest'
```bash
# uv 사용
uv sync --extra dev
uv run pytest tests/ -v

# 또는 pip 사용
pip install pytest pytest-cov
pytest tests/ -v
```

#### ModuleNotFoundError: No module named 'folder_paths'
**정상 동작**: 이는 의도된 동작입니다. `tests/conftest.py`가 자동으로 mock 처리합니다.

#### GPU 테스트가 skip되는 경우
```bash
# CUDA 사용 가능 여부 확인
python -c "import torch; print(torch.cuda.is_available())"

# False라면:
# - CUDA 드라이버 설치 확인
# - PyTorch CUDA 버전 재설치
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### 통합 테스트가 skip되는 경우
```bash
# 테스트 영상 확인
ls -lh examples/vfi_test_*.mp4

# 없다면 영상 준비 필요
# examples/ 디렉토리에 vfi_test_*.mp4 파일 배치
```

### 특정 테스트만 디버깅

```bash
# 실패한 테스트만 재실행
pytest tests/ --lf -v

# 특정 테스트 하나만 실행
pytest tests/test_frame_pair_slicer.py::TestFramePairSlicerBasics::test_input_types_structure -v

# PDB 디버거로 실행
pytest tests/test_frame_pair_slicer.py -v --pdb

# 출력 캡처 비활성화 (print 확인)
pytest tests/ -v -s
```

## 📈 고급 사용법

### 병렬 테스트 실행 (빠른 실행)

```bash
# pytest-xdist 설치
pip install pytest-xdist

# CPU 코어 수만큼 병렬 실행
pytest tests/ -n auto

# 4개 프로세스로 실행
pytest tests/ -n 4
```

### 특정 Python 버전 테스트

```bash
# Python 3.11로 테스트
python3.11 -m pytest tests/ -v

# 가상환경 사용
python3.11 -m venv venv311
source venv311/bin/activate  # Linux/macOS
# venv311\Scripts\activate  # Windows
pip install -r requirements.txt
pytest tests/ -v
```

### 테스트 결과 JUnit XML로 내보내기

```bash
# CI/CD 통합용
pytest tests/ --junitxml=test-results.xml
```

## 📚 추가 자료

### pytest 마커 목록

현재 프로젝트에서 사용하는 마커:

- `@pytest.mark.unit` - 단위 테스트
- `@pytest.mark.integration` - 통합 테스트
- `@pytest.mark.slow` - 느린 테스트 (1분 이상)
- `@pytest.mark.requires_gpu` - GPU 필수 테스트
- `@pytest.mark.requires_model` - 모델 파일 필수 테스트

### 파일 구조

```
tests/
├── README.md                    # 이 파일
├── conftest.py                  # pytest 설정 및 fixtures
├── test_frame_pair_slicer.py    # FramePairSlicer 단위 테스트
├── test_node_registration.py   # 노드 등록 테스트
└── test_integration.py          # 통합 테스트
```

### 유용한 pytest 옵션

```bash
# 가장 느린 10개 테스트 표시
pytest tests/ --durations=10

# 테스트 실행 순서 무작위화
pytest tests/ --random-order

# 첫 실패 시 중단
pytest tests/ -x

# 최대 2개 실패까지만 허용
pytest tests/ --maxfail=2

# 경고 표시
pytest tests/ -v -W all
```

## 🎯 RTX 4090 테스트 체크리스트

GPU 머신에서 전체 테스트를 실행하기 전 확인사항:

- [ ] CUDA 드라이버 설치 확인 (`nvidia-smi`)
- [ ] PyTorch CUDA 지원 확인 (`torch.cuda.is_available()`)
- [ ] 의존성 설치 (`pip install -r requirements.txt`)
- [ ] 테스트 영상 파일 존재 확인 (`ls examples/vfi_test_*.mp4`)
- [ ] 전체 테스트 실행 (`pytest tests/ -v`)
- [ ] GPU 테스트 통과 확인 (0 skipped for GPU tests)
- [ ] 커버리지 측정 (`pytest tests/ --cov=nodes --cov=utils`)

### 예상 테스트 시간

| 환경 | 단위 테스트 | 통합 테스트 | 전체 |
|------|------------|------------|------|
| CPU (CI) | ~2-3초 | ~1-2초 | ~3-5초 |
| RTX 4090 | ~2-3초 | ~1-2초 | ~3-5초 |

*GPU 테스트는 텐서 연산만 하므로 시간 차이가 거의 없음*

## 📞 문제 보고

테스트 관련 문제 발견 시:
1. GitHub Issues에 보고: https://github.com/Rockheung/ComfyUI-TLBVFI-TF32/issues
2. 다음 정보 포함:
   - OS 및 Python 버전
   - GPU 정보 (해당 시)
   - pytest 출력 전문
   - `pip list` 출력
