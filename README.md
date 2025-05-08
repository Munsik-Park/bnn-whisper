# BNN-Whisper

Bayesian Neural Network (BNN)을 활용한 Whisper 모델 기반 음성 인식 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 OpenAI의 Whisper 모델을 Bayesian Neural Network로 변환하여 음성 인식의 불확실성을 정량화하고, 더 신뢰할 수 있는 음성 인식 결과를 제공하는 것을 목표로 합니다.

## 주요 기능

- Whisper 모델의 BNN 변환
- 음성 인식 결과의 불확실성 추정
- 다양한 언어 지원
- 실시간 음성 인식 지원

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/yourusername/bnn-whisper.git
cd bnn-whisper
```

2. 가상환경 생성 및 활성화
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows의 경우:
venv\Scripts\activate
# macOS/Linux의 경우:
source venv/bin/activate
```

3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

1. 가상환경 활성화
```bash
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

2. 음성 파일 인식
```python
from src.bnn_whisper import BNNWhisper

model = BNNWhisper()
result = model.transcribe("data/audio/your_audio.wav")
print(result)
```

3. 불확실성 추정
```python
result, uncertainty = model.transcribe_with_uncertainty("data/audio/your_audio.wav")
print(f"인식 결과: {result}")
print(f"불확실성: {uncertainty}")
```

## 라이선스

MIT License