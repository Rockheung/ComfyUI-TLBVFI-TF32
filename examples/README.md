# Test Video Files

이 디렉토리에는 ComfyUI-TLBVFI-TF32 통합 테스트용 샘플 영상이 포함되어 있습니다.

## 📹 영상 정보

### 출처
- **원본 영상**: [Big Buck Bunny - Blender Foundation](https://www.youtube.com/watch?v=QiKpjrGK9Uw)
- **구간**: 3:40 ~ 3:50 (약 10초)
- **라이선스**: Creative Commons Attribution 3.0
- **제작**: Blender Foundation (2008)

### 파일 목록

| 파일명 | 해상도 | 크기 | 용도 |
|--------|--------|------|------|
| `vfi_test_360p.mp4` | 360p (640×360) | ~1.5 MB | 빠른 테스트 |
| `vfi_test_480p.mp4` | 480p (854×480) | ~2.4 MB | 기본 테스트 |
| `vfi_test_720p.mp4` | 720p (1280×720) | ~4.6 MB | HD 테스트 |
| `vfi_test_1080p.mp4` | 1080p (1920×1080) | ~7.2 MB | Full HD 테스트 |
| `vfi_test_1440p.mp4` | 1440p (2560×1440) | ~14.3 MB | QHD 테스트 |
| `vfi_test_4K.webm` | 4K (3840×2160) | ~13.4 MB | 4K 테스트 |

**총 크기**: ~43 MB

## 🎯 사용 목적

### 1. 통합 테스트
```bash
pytest tests/test_integration.py -v
```

테스트 항목:
- 파일 존재 확인
- 파일 읽기 가능 여부
- 파일 크기 검증
- 실제 해상도로 노드 동작 검증

### 2. 수동 테스트

ComfyUI에서 실제 프레임 보간 품질 확인:
1. ComfyUI 실행
2. VHS LoadVideo로 테스트 영상 로드
3. TLBVFI 노드로 프레임 보간
4. 결과 확인

### 3. 벤치마킹

다양한 해상도에서 처리 시간 측정:
- 360p: 빠른 프로토타입 테스트
- 1080p: 실사용 시나리오
- 4K: 성능 한계 테스트

## 📝 라이선스

이 테스트 영상들은 [Big Buck Bunny](https://peach.blender.org/) 프로젝트에서 발췌되었으며,
**Creative Commons Attribution 3.0** 라이선스를 따릅니다.

### 원본 저작권 정보

```
(c) Copyright 2008, Blender Foundation / www.bigbuckbunny.org
```

### 라이선스 내용

이 영상을 다음과 같이 자유롭게 사용할 수 있습니다:
- ✅ 상업적 사용 가능
- ✅ 수정 및 재배포 가능
- ✅ 2차 저작물 제작 가능

**조건**: 원작자 표시 (Attribution)

자세한 내용: https://creativecommons.org/licenses/by/3.0/

## 🔗 관련 링크

- [Big Buck Bunny 공식 사이트](https://peach.blender.org/)
- [YouTube 원본 영상](https://www.youtube.com/watch?v=QiKpjrGK9Uw)
- [Blender Foundation](https://www.blender.org/about/foundation/)
- [Creative Commons BY 3.0 License](https://creativecommons.org/licenses/by/3.0/)

## 📊 영상 사양

### 공통 사양
- **코덱**: H.264 (MP4) / VP9 (WebM)
- **프레임 레이트**: 30 fps
- **길이**: ~10초
- **오디오**: 없음 (영상만)
- **인코딩**: 고품질 (테스트 정확도 확보)

### 추출 명령 예시

원본 영상에서 테스트 클립 생성 시 사용한 명령:

```bash
# 360p
ffmpeg -i source.mp4 -ss 03:40 -t 10 -vf scale=640:360 -c:v libx264 -crf 18 -an vfi_test_360p.mp4

# 1080p
ffmpeg -i source.mp4 -ss 03:40 -t 10 -vf scale=1920:1080 -c:v libx264 -crf 18 -an vfi_test_1080p.mp4

# 4K (WebM)
ffmpeg -i source.mp4 -ss 03:40 -t 10 -vf scale=3840:2160 -c:v libvpx-vp9 -crf 18 -an vfi_test_4K.webm
```

## ⚠️ 주의사항

### Git 저장소 크기
- 이 영상 파일들로 인해 레포지토리 크기가 ~43MB 증가합니다
- 필요하지 않다면 로컬에서 삭제 가능:
  ```bash
  rm examples/vfi_test_*.mp4 examples/vfi_test_*.webm
  ```
- 통합 테스트는 영상이 없으면 자동으로 skip됩니다

### 대체 영상 사용
다른 테스트 영상을 사용하려면:
1. `examples/` 디렉토리에 `vfi_test_` prefix로 파일 배치
2. MP4 또는 WebM 형식
3. 통합 테스트가 자동으로 인식

## 🙏 감사의 말

Blender Foundation의 [Big Buck Bunny](https://peach.blender.org/) 프로젝트에 감사드립니다.
오픈 소스 3D 애니메이션 영화를 제공해주셔서 테스트에 활용할 수 있었습니다.
