# BNN-Whisper

Bayesian Neural Network (BNN)을 활용한 Whisper 모델 기반 음성 인식 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 OpenAI의 Whisper 모델을 Bayesian Neural Network로 변환하여 음성 인식의 불확실성을 정량화하고, 더 신뢰할 수 있는 음성 인식 결과를 제공하는 것을 목표로 합니다. 특히 한국어 음성 인식에 최적화되어 있습니다.

## 주요 기능

- Whisper 모델의 BNN 변환
- 음성 인식 결과의 불확실성 추정
- 실시간 음성 녹음 및 인식
- CPU/MPS 성능 비교 및 모니터링
- 메모리 사용량 추적

## 성능 테스트 결과

### CPU vs MPS (Apple Silicon) 비교
- 처리 시간:
  - CPU: 6.27초 (전처리: 0.01초, 추론: 6.24초)
  - MPS: 2.90초 (전처리: 0.00초, 추론: 2.89초)
  - 성능 향상률: MPS가 CPU보다 2.16배 빠름

### 메모리 사용량
- CPU: 749.56 MB
- MPS: 892.48 MB

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

## 프로젝트 구조

## 사용 방법

1. 가상환경 활성화
```bash
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

2. 음성 인식 테스트 실행
```bash
python tests/test_korean.py
```

3. 테스트 실행 시:
   - Enter를 눌러 녹음 시작
   - 다시 Enter를 눌러 녹음 중지
   - CPU와 MPS 각각의 성능 결과 확인

## 주요 특징

- 실시간 음성 녹음 및 인식
- 불확실성 측정을 통한 신뢰도 평가
- CPU/MPS 성능 비교 기능
- 메모리 사용량 모니터링
- 한국어 음성 인식 최적화

## 시스템 요구사항

- Python 3.10 이상
- PyTorch 2.2.2 이상
- macOS (MPS 지원) 또는 Linux/Windows (CPU/CUDA 지원)

## 라이선스

MIT License