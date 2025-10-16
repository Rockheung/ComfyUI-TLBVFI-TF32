# TLBVFI Chunk 기반 인터폴레이션 전략 분석

## 1. 목표
- **해상도/길이**: 4K 60fps, 10분 이상 (약 36,000 프레임)
- **안정성**: GPU OOM 없이 장시간 실행
- **파이프라인**: 프레임별 보간 결과를 즉시 디스크에 청크 동영상으로 인코딩 후 메모리 해제
- **결합 방식**: HLS처럼 손실 없이(=재인코딩 없이) 최종 영상으로 통합 가능

## 2. 제안된 방식 개요
1. 원본 영상을 프레임 쌍 단위로 순회하며 보간 프레임을 생성
2. `(prev_frame, interpolated_frames…)`를 묶어 짧은 청크 영상으로 인코딩
3. 청크가 저장되면 GPU/CPU 메모리에서 관련 텐서를 즉시 해제
4. 모든 청크 생성 후, 같은 코덱/설정으로 인코딩된 파일들을 무손실 concat

## 3. 기존 구현과의 비교
| 항목 | 제안 방식 | 현재 저장소 구현 (ChunkVideoSaver + VideoConcatenator) |
|------|-----------|---------------------------------------------------------|
| 프레임 순회 | 프레임별 수동 루프 | `TLBVFI_FramePairSlicer` / `TLBVFI_ChunkProcessor`가 자동 처리 |
| 청크 저장 | ffmpeg 기반 직접 구현 필요 | `ChunkVideoSaver` 노드가 MP4 청크 생성 |
| 메모리 관리 | 보간 직후 수동 해제 | `cpu_offload`, `soft_empty_cache`, 주기적 clear 로 자동 처리 |
| 최종 합치기 | HLS 스타일 concat 계획 | `VideoConcatenator`가 ffmpeg concat demuxer로 무손실 병합 |

**결론**: 제안 시나리오는 이미 리포지터리에 포함된 "청크 기반 파이프라인"과 거의 동일합니다.  
- `ChunkVideoSaver`는 각 프레임 쌍 결과(원본 프레임 + 보간 프레임)를 NVENC/H.264 등으로 인코딩한 MP4 파일을 출력
- `VideoConcatenator`는 `-c copy` 방식으로 재인코딩 없이 MP4 청크를 병합
- `TLBVFI_ChunkProcessor` 노드는 위 과정을 한 번에 실행하는 자동 루프(세션 관리 포함)이며, 내부적으로 `TLBVFI_Interpolator_V2`를 사용해 TF32/flow scale/sample steps 옵션을 그대로 지원한다.
- 루프 노드를 사용할 수 없는 ComfyUI 코어 환경에서는 `TLBVFI_BatchInterpolator_V2`가 같은 순회 로직을 내부적으로 처리한다.

## 4. 타당성 분석
### 4.1 메모리 관점
- 인터폴레이터는 한 번에 두 프레임(±보간 결과)만 GPU에 올리므로 VRAM 사용량은 **모델(≈3.6GB) + 입력 텐서(≈300MB)** 수준으로 유지
- 청크로 즉시 디스크에 기록하면 GPU/CPU 쪽 텐서를 즉시 free 가능 → 장시간 작업 시 누적 메모리 문제 최소화
- 저장소 구현은 이미 `cpu_offload`, `torch.cuda.empty_cache()` 호출, 10페어마다 soft cache clear 등을 포함

### 4.2 디스크 I/O 및 성능
- 4K 60fps 영상 10분(36,000프레임)을 2배 보간하면 프레임 수가 71,999로 늘어남 → 청크당 포함 프레임 수, 인코딩 설정에 따라 총 파일 용량은 입력 대비 1.5~3배까지 증가할 수 있음
- NVENC 사용 시 실시간(또는 그 이상) 인코딩이 가능, CPU 인코딩일 경우 속도 저하 가능 → `codec` 옵션으로 환경에 맞는 하드웨어 인코더 선택 권장
- SSD 사용, 청크 크기(예: 8~32프레임)를 적절히 조절하면 I/O 병목을 피할 수 있음

### 4.3 무손실 concat
- ffmpeg concat demuxer는 **동일한 인코딩 파라미터**(코덱, 프로파일, 픽셀 포맷, GOP 구조 등)를 가진 MP4/H.264 청크만 무손실로 합칠 수 있음
- `ChunkVideoSaver`는 청크 생성 시 동일한 설정을 유지하도록 설계되어 concat 시 재인코딩이 필요 없음
- 완전한 비손실을 원하면 `codec`을 `ffv1` 등 무손실 코덱으로 선택하거나, PNG 프레임 + 별도 mux 방식을 고려해야 하지만 파일 용량/속도가 크게 증가

## 5. 구현 권장사항
1. **워크플로 선택**
   - 수동 구성(비추천): `VHS_LoadVideo → TLBVFI_FramePairSlicer → TLBVFI_Interpolator_V2 → ChunkVideoSaver → VideoConcatenator`
   - 자동 전체 처리: `TLBVFI_ChunkProcessor` 이용 (권장, 작업 중단 후 재시작/세션 관리 지원)
   - 루프 노드를 사용할 수 없는 경우: `VHS_LoadVideo → TLBVFI_BatchInterpolator_V2 → ChunkVideoSaver → VideoConcatenator`  
     (`TLBVFI_BatchInterpolator_V2`가 내부 루프로 전체 페어를 보간하며, V2의 최신 옵션을 모두 제공)
2. **청크 크기 조절**
   - `ChunkVideoSaver`의 `chunk_id`를 프레임 단위로 증가시키면 청크 하나에 모든 결과가 들어가므로, 필요 시 세션당 프레임 수 기반으로 chunk rotation을 구현 (현재 Saver는 1 pair = 1 chunk 구조)
3. **메모리 옵션**
   - `enable_tf32=True`, `cpu_offload=True` 유지 → 4K 60fps에서도 8~10GB VRAM 환경에서 안전
   - CPU 오프로딩이 꺾을 때 처리 속도 저하 가능 → VRAM이 충분한 RTX4090(24GB)에서는 `cpu_offload=False`도 고려 가능
4. **인코딩 세부 설정**
   - `codec`: `h264_nvenc`, `hevc_nvenc` 등 GPU 가속 인코더 권장
   - `crf`: 품질/용량 트레이드오프 (18은 visually lossless 수준)
   - `pix_fmt`: `yuv420p`가 호환성 최고, 필요 시 `yuv444p` 등 변경 가능 (concat 시 모든 청크 동일해야 함)
5. **장시간 작업 모니터링**
   - 세션 폴더(`output/tlbvfi_chunks/<session_id>`) 용량 확인
   - ffmpeg 로그 모니터링 (권장: saver 노드에 예외 처리/로그 보강)

## 6. 추가 개선 아이디어
- **Adaptive chunk sizing**: 프레임 수나 디스크 용량에 따라 청크를 묶는 정책 제공
- **FFmpeg 파이프라인 개선**: NVENC 파라미터 튜닝(lookahead, bframes 등)으로 품질/속도 최적화
- **메타데이터**: 청크별 manifest에 프레임 범위/타임스탬프 저장 → concat 후 검증 가능
- **모니터링 유틸**: 처리 진행률, 예상 완료 시간, VRAM/디스크 사용량을 노드 UI로 출력

## 7. 루프 노드 없이 구성하는 배치 워크플로

ComfyUI 코어 빌드에는 순회 루프 노드가 포함되어 있지 않지만, `TLBVFI_BatchInterpolator_V2`를 사용하면 동일한 자동 순회를 노드 내부에서 처리할 수 있다.  
이 노드는 `TLBVFI_Interpolator_V2`의 최신 로직을 재사용하지만, 모든 프레임을 하나의 텐서로 반환하기 때문에 **긴 영상에서는 RAM 사용량이 매우 커진다**는 점을 주의해야 한다.

### 그래프 예시
`VHS_LoadVideo → TLBVFI_BatchInterpolator_V2 → PreviewImage (옵션) → ChunkVideoSaver → VideoConcatenator`

- **TLBVFI_BatchInterpolator_V2**: 입력 `IMAGE` 텐서를 받아 인접 프레임 페어를 자동 보간한다. 다만 결과 전체를 메모리에 보관한 채 반환하므로 수천 프레임 이상의 영상에는 적합하지 않다.
- **ChunkVideoSaver**: 배치 노드가 반환한 시퀀스를 받아 지정한 인코더와 CRF로 MP4를 저장한다. 이 경로를 사용할 경우 ChunkVideoSaver가 한 번에 큰 텐서를 처리하게 된다.
- **VideoConcatenator**: 동일한 세션의 manifest를 읽어 재인코딩 없이 최종 영상을 합친다.

### 활용 가이드
- 짧은 테스트 클립이나 저해상도 영상에는 간단하게 적용 가능하다.
- 장편·고해상도 영상에서는 VRAM/시스템 RAM 누수를 피하려면 `TLBVFI_ChunkProcessor` 경로(청크 단위로 즉시 디스크 저장)를 사용하는 것이 안전하다.
- Queue Prompt와 조합해 여러 영상을 순차 처리할 때에도, 대용량 작업은 ChunkProcessor 기반 파이프라인을 권장한다.

## 8. 결론
- 사용자 제안 방식은 이미 저장소에 포함된 청크 파이프라인과 동일한 핵심 아이디어를 공유하며, 4K 60fps 10분 이상의 영상 처리에도 충분히 타당하다.
- `TLBVFI_ChunkProcessor + ChunkVideoSaver + VideoConcatenator` 조합을 활용하면, 메모리 안정성·무손실 concat·자동 세션 관리까지 지원하므로 별도 재구현 없이 목표 달성이 가능하다. ChunkProcessor는 이제 Interpolator V2 로직을 재사용해 TF32/flow/sample 옵션을 그대로 제공한다.
- 루프 노드가 없는 환경에서는 `TLBVFI_BatchInterpolator_V2` 기반 워크플로로 동일한 순회 구조를 확보하면서도 V2 옵션을 그대로 활용할 수 있다.
- 필요한 경우 상기 개선 아이디어를 추가해 장시간 안정성과 모니터링을 강화할 수 있다.
