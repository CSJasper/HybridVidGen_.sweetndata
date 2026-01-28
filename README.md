# Beat-Synchronized Hybrid Video Generator

### 1. beat_hybrid_vidgen.py

**작동 방식:**
- 각 peak 클립을 개별적으로 생성하는 방식입니다
- 여러 개의 독립적인 T2V 생성 작업을 배치로 실행합니다
- 각 클립은 서로 다른 시드를 사용하여 다양한 영상을 생성합니다
- 생성된 클립 풀에서 랜덤하게 선택하여 타임라인에 배치합니다

**장점:**
- 각 클립이 독립적으로 생성되어 더 다양한 시각적 표현이 가능
- 특정 클립 생성 실패 시 다른 클립에 영향을 주지 않음

**단점:**
- 전체 생성 시간이 더 오래 걸릴 수 있음
- 각 클립 간 시각적 일관성이 다소 떨어질 수 있음

### 2. beat_hybrid_vidgen_sliced.py

**작동 방식:**
- 하나의 긴 비디오를 생성한 후 여러 세그먼트로 슬라이싱하는 방식입니다
- 단일 T2V 생성으로 필요한 모든 peak 클립의 총 길이만큼의 비디오를 생성합니다
- 생성된 긴 비디오를 동일한 길이의 클립으로 나눕니다
- 랜덤 순서로 셔플하여 타임라인에 배치합니다

**장점:**
- T2V 모델을 한 번만 로드하므로 전체 생성 시간이 단축될 수 있음
- 연속된 비디오에서 나온 클립들이므로 시각적 일관성이 높음

**단점:**
- 단일 시드로 생성되므로 시각적 다양성이 상대적으로 제한적
- 한 번의 생성 실패 시 모든 클립에 영향을 줄 수 있음

## 설치 방법

### 필요한 라이브러리 세팅

```bash
conda create -n hybrid_vidgen python=3.11 -y
conda activate hybrid_vidgen
pip install -r requirements.txt --no-deps
```

### 추가 설정

1. BiM-VFI 클론:
```bash
git clone https://github.com/KAIST-VICLab/BiM-VFI.git
```

2. HuggingFace 인증 (Wan 모델 사용 시):
```bash
huggingface-cli login
```

## 사용 방법

### 기본 사용법

```bash
python beat_hybrid_vidgen.py --audio music.mp3 --output output.mp4
```

또는

```bash
python beat_hybrid_vidgen_sliced.py --audio music.mp3 --output output.mp4
```

### 주요 옵션 설명

#### 필수 옵션
- `--audio AUDIO`: 입력 오디오 파일 경로 (필수)
- `--output OUTPUT`: 출력 비디오 파일 경로 (필수)

#### 비디오 생성 옵션
- `--model MODEL`: Wan T2V 모델 ID (기본값: "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
- `--width WIDTH`: 비디오 너비 (기본값: 832)
- `--height HEIGHT`: 비디오 높이 (기본값: 480)
- `--num-peak-clips N`: 생성할 peak 클립 수 (기본값: 10)
- `--peak-len SECONDS`: 각 peak 클립 길이 (기본값: 1.5초)
- `--batch-size N`: 배치 생성 크기 (기본값: 1)

#### 보간 모드 옵션
- `--interpolation-mode MODE`: 보간 방식 선택
  - `hybrid`: 긴 전환은 FLF2V, 짧은 전환은 BiM-VFI 사용 (기본값)
  - `flf2v`: 모든 전환에 FLF2V 사용 (고품질, 느림)
  - `bim`: 모든 전환에 BiM-VFI 사용 (빠름, 중간 품질)
- `--hybrid-threshold SECONDS`: hybrid 모드에서 FLF2V/BiM-VFI 전환 기준 (기본값: 1.5초)
- `--max-flf2v-count N`: 최대 FLF2V 사용 횟수 (-1은 무제한, 기본값: -1)
- `--flf2v-model MODEL`: FLF2V 모델 ID (기본값: "Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers")

#### 프롬프트 옵션
- `--prompt-style STYLE`: 프롬프트 스타일 선택
  - `abstract`: 추상적 비주얼 (기본값)
  - `nature`: 자연 풍경
  - `cosmic`: 우주/코스믹
  - `urban`: 도시 풍경
  - `fantasy`: 판타지
  - `particle`: 파티클 효과
- `--peak-prompt TEXT`: peak 클립용 커스텀 프롬프트
- `--transition-prompt TEXT`: transition 클립용 커스텀 프롬프트

#### 비트 감지 옵션
- `--peak-thresh THRESHOLD`: peak 감지 임계값 (기본값: 0.5)

#### 오디오 범위 옵션
- `--audio-start SECONDS`: 오디오 시작 지점 (기본값: 0)
- `--audio-end SECONDS`: 오디오 종료 지점 (기본값: None, 끝까지)

#### 성능 최적화 옵션
- `--cpu-offload MODE`: CPU 오프로드 모드
  - `model`: 모델 단위 오프로드 (빠름, 기본값)
  - `sequential`: 순차적 오프로드 (메모리 절약)
  - `none`: 오프로드 없음 (최고 성능)
- `--device-map MAP`: 멀티 GPU 설정
  - `auto`: 자동 분산
  - `balanced`: 균형 분산
  - None: 단일 GPU (기본값)
- `--hf-token TOKEN`: HuggingFace API 토큰
- `--cache-dir PATH`: HuggingFace 캐시 디렉토리
- `--local-files-only`: 로컬 캐시만 사용

## 사용 예시

### 기본 비디오 생성
```bash
python beat_hybrid_vidgen.py \
  --audio music.mp3 \
  --output output.mp4 \
  --num-peak-clips 15 \
  --peak-len 2.0
```

### 우주 테마 비디오 (FLF2V 사용)
```bash
python beat_hybrid_vidgen_sliced.py \
  --audio music.mp3 \
  --output cosmic_video.mp4 \
  --prompt-style cosmic \
  --interpolation-mode flf2v \
  --width 1280 \
  --height 720
```

### 빠른 생성 (BiM-VFI만 사용)
```bash
python beat_hybrid_vidgen.py \
  --audio music.mp3 \
  --output fast_video.mp4 \
  --interpolation-mode bim \
  --cpu-offload sequential
```

### 오디오 특정 구간만 처리
```bash
python beat_hybrid_vidgen.py \
  --audio music.mp3 \
  --output chorus_video.mp4 \
  --audio-start 60 \
  --audio-end 120
```

### 커스텀 프롬프트 사용
```bash
python beat_hybrid_vidgen_sliced.py \
  --audio music.mp3 \
  --output custom_video.mp4 \
  --peak-prompt "cinematic neon lights in a cyberpunk city" \
  --transition-prompt "smooth morphing between scenes"
```

## 출력 구조

생성된 비디오와 함께 다음과 같은 중간 파일들이 생성됩니다:

```
work_YYYYMMDD_HHMMSS/
├── peak_0000.mp4          # 생성된 peak 클립들
├── peak_0001.mp4
├── ...
├── transition_0000.mp4    # 생성된 transition 클립들
├── transition_0001.mp4
├── ...
└── segments_processed/    # 어셈블리용 처리된 세그먼트
```

## 기술 세부사항

### 비트 감지
- Madmom의 RNN 기반 비트 트래커 사용
- Downbeat와 regular beat 구분
- 비트 강도(strength) 분석

### 비디오 생성
- **Peak 클립**: Wan T2V 모델로 생성된 고품질 영상
- **Transition 클립**: 두 가지 방식으로 생성
  - **FLF2V**: Wan 2.1의 First-Last-Frame to Video (고품질, 느림)
  - **BiM-VFI**: Bilateral Motion Video Frame Interpolation (빠름, 중간 품질)

### 하이브리드 모드
- 전환 구간의 길이에 따라 자동으로 최적의 방법 선택
- 긴 전환 (>= threshold): FLF2V 사용
- 짧은 전환 (< threshold): BiM-VFI 사용

### 루프 전환
- 비디오의 시작과 끝을 자연스럽게 연결
- 완벽한 루핑 비디오 생성 가능

## 라이선스

이 프로젝트에서 사용하는 주요 라이브러리들의 라이선스를 확인하세요:
- Wan AI 모델: 각 모델의 HuggingFace 페이지 참조
- BiM-VFI: 해당 레포지토리의 라이선스 참조
- Madmom: BSD License

## 문제 해결

### 메모리 부족
```bash
# CPU 오프로드 모드 사용
--cpu-offload sequential

# 배치 크기 줄이기
--batch-size 1

# 비디오 해상도 낮추기
--width 640 --height 360
```

### 생성 속도 개선
```bash
# BiM-VFI만 사용
--interpolation-mode bim

# 캐시 디렉토리를 SSD로 설정
--cache-dir /path/to/ssd/cache
```

### NumPy 경고
```bash
# NumPy 1.24 이상 설치
pip install 'numpy>=1.24.0'
```
