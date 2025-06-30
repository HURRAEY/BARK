# 🎭 감정별 Bark TTS 파인튜닝

Apple Silicon (MPS)에서 감정별 음성 데이터를 활용한 Bark TTS 파인튜닝 프로젝트입니다.

## 📁 프로젝트 구조

```
BARK/
├── 일어/                          # 감정별 음성 데이터
│   ├── 우울/
│   │   └── added_B.LJJ.JP30m_v2.zip
│   ├── 슬픔/
│   │   └── S.LJJ.JP30m_600e_58200s.zip
│   └── 기쁨/
│       └── added_J.LJJ.JP30m_v2.zip
├── finetune_bark_emotions.py      # 메인 파인튜닝 스크립트
├── test_emotion_voices.py         # 테스트 스크립트
├── requirements_emotion.txt       # 의존성 파일
└── README_emotion_finetune.md     # 이 파일
```

## 🚀 설치 및 설정

### 1. 의존성 설치

```bash
# 가상환경 생성 (권장)
python -m venv venv_emotion
source venv_emotion/bin/activate  # macOS/Linux
# 또는 Windows: venv_emotion\Scripts\activate

# 의존성 설치
pip install -r requirements_emotion.txt
```

### 2. 환경 설정

Apple Silicon Mac에서 MPS 사용을 위한 환경변수가 스크립트에 자동으로 설정됩니다:

```python
os.environ["SUNO_ENABLE_MPS"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "False"
os.environ["SUNO_OFFLOAD_CPU"] = "False"
```

## 🎯 사용법

### 1. 전체 감정 파인튜닝

```bash
# 모든 감정(우울, 슬픔, 기쁨)을 순차적으로 파인튜닝
python finetune_bark_emotions.py --emotion all
```

### 2. 특정 감정만 파인튜닝

```bash
# 우울 감정만
python finetune_bark_emotions.py --emotion 우울

# 슬픔 감정만
python finetune_bark_emotions.py --emotion 슬픔

# 기쁨 감정만
python finetune_bark_emotions.py --emotion 기쁨
```

### 3. 커스텀 데이터 디렉토리 지정

```bash
python finetune_bark_emotions.py --emotion all --data_dir "내_데이터_폴더"
```

## 🧪 테스트

파인튜닝이 완료된 후 감정별 음성을 테스트할 수 있습니다:

```bash
# 모든 감정 테스트
python test_emotion_voices.py

# 특정 감정만 테스트
python test_emotion_voices.py --emotions 우울 기쁨
```

## 📊 파인튜닝 프로세스

### 1. 데이터 추출

- ZIP 파일에서 오디오 데이터 자동 추출
- 지원 포맷: WAV, MP3, FLAC, M4A

### 2. 데이터 전처리

- 24kHz 샘플레이트로 리샘플링
- 1-30초 길이 오디오만 선택
- 처음 50개 파일만 처리 (속도 최적화)

### 3. 음성 프리셋 생성

- 가장 긴 오디오를 대표 샘플로 선택
- Bark 호환 프리셋 형태로 저장

### 4. 감정별 설정

| 감정 | 화자 ID            | Temperature          | 감정 태그                        |
| ---- | ------------------ | -------------------- | -------------------------------- |
| 우울 | depression_speaker | text: 0.6, wave: 0.7 | [sad], [melancholy], [depressed] |
| 슬픔 | sadness_speaker    | text: 0.5, wave: 0.6 | [crying], [sorrow], [grief]      |
| 기쁨 | joy_speaker        | text: 0.7, wave: 0.8 | [happy], [excited], [joyful]     |

## 📁 출력 구조

파인튜닝 후 다음과 같은 구조로 결과물이 생성됩니다:

```
finetune_output/
├── 우울/
│   ├── extracted/              # 추출된 원본 데이터
│   ├── processed_audio/        # 전처리된 오디오
│   ├── voice_presets/          # Bark 음성 프리셋
│   │   └── depression_speaker/
│   │       └── speaker.wav
│   ├── generated_samples/      # 테스트 생성 샘플
│   └── metadata.json          # 데이터셋 메타정보
├── 슬픔/
│   └── ... (동일 구조)
└── 기쁨/
    └── ... (동일 구조)
```

## 🎤 테스트 결과

테스트 실행 후 다음 위치에 결과물이 생성됩니다:

```
emotion_test_results/
├── 우울/
│   ├── 우울_tagged_1.wav      # 감정 태그 포함 버전
│   ├── 우울_preset_1.wav      # 프리셋만 사용 버전
│   └── ...
├── 슬픔/
├── 기쁨/
└── comparison/                 # 같은 문장 다른 감정 비교
    ├── comparison_우울.wav
    ├── comparison_슬픔.wav
    └── comparison_기쁨.wav
```

## ⚠️ 주의사항

### 메모리 요구사항

- **최소**: 16GB RAM (소형 모델)
- **권장**: 32GB+ RAM (풀사이즈 모델)

### 처리 시간

- M1 Pro 기준 감정당 약 30-60분 소요
- GPU 가속 사용 시 상당한 속도 향상

### 데이터 품질

- 고품질 오디오 데이터 사용 권장
- 일관된 화자와 녹음 환경 필요
- 전사(transcript) 데이터가 있으면 더 좋은 결과

## 🔧 문제 해결

### 1. MPS 관련 오류

```bash
# CPU 모드로 폴백
export SUNO_ENABLE_MPS=False
```

### 2. 메모리 부족

```bash
# 소형 모델 사용
export SUNO_USE_SMALL_MODELS=True
```

### 3. ZIP 파일 인식 안됨

- 파일 경로와 이름 확인
- 압축 파일 손상 여부 확인

## 📈 성능 최적화

### Apple Silicon 최적화

- MPS 백엔드 사용으로 GPU 가속
- 통합 메모리 활용
- 배치 크기 조정으로 메모리 효율성 향상

### 품질 향상 팁

1. **더 많은 데이터**: 50개 → 전체 데이터 사용
2. **전사 데이터**: 정확한 텍스트 제공
3. **데이터 정제**: 노이즈 제거, 길이 조정
4. **Temperature 조정**: 감정별 최적값 찾기

## 🎯 향후 개선 사항

- [ ] 실제 전사 데이터 활용
- [ ] 더 정교한 감정 분류
- [ ] 실시간 음성 변환
- [ ] 웹 인터페이스 추가
- [ ] 다국어 지원 확장

## 📚 참고 자료

- [Bark GitHub](https://github.com/suno-ai/bark)
- [Apple Silicon PyTorch](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)
- [음성 감정 인식 연구](https://arxiv.org/abs/2209.03143)

---

🎭 **Happy Fine-tuning!** 감정이 풍부한 AI 음성을 만들어보세요!
