import torch
import torch.nn as nn
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from bayesian_torch.layers import LinearReparameterization
import numpy as np
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BNNWhisper(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/whisper-base",
        device: Optional[str] = None,
        num_samples: int = 10,
        default_language: str = "ko"  # 기본 언어를 한국어로 설정
    ):
        """
        BNN-Whisper 모델 초기화
        
        Args:
            model_name (str): 사용할 Whisper 모델 이름
            device (str, optional): 사용할 디바이스 ('cuda' 또는 'cpu')
            num_samples (int): Monte Carlo 샘플링 횟수
            default_language (str): 기본 언어 설정 (기본값: 'ko' - 한국어)
        """
        super().__init__()
        self.model_name = model_name
        self.num_samples = num_samples
        self.default_language = default_language
        
        # 디바이스 설정
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Whisper 모델 및 프로세서 로드
        logger.info(f"Loading Whisper model: {model_name}")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.whisper = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.whisper.to(self.device)
        
        # BNN 변환은 일단 비활성화
        # self._convert_to_bnn()
        
        logger.info(f"Model initialized with default language: {default_language}")
    
    def _preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        오디오 파일 전처리
        
        Args:
            audio_path (str): 오디오 파일 경로
            
        Returns:
            torch.Tensor: 전처리된 오디오 텐서
        """
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        return waveform.squeeze()
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Tuple[str, float]:
        """
        음성 파일을 텍스트로 변환
        
        Args:
            audio_path (str): 오디오 파일 경로
            language (str, optional): 언어 코드 (기본값: 'ko' - 한국어)
            task (str): 작업 유형 ('transcribe' 또는 'translate')
            
        Returns:
            Tuple[str, float]: (인식 결과, 불확실성)
        """
        # 언어 설정 (기본값: 한국어)
        language = language or self.default_language
        
        # 오디오 전처리
        waveform = self._preprocess_audio(audio_path)
        
        # 입력 특성 추출
        input_features = self.processor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Monte Carlo 샘플링을 통한 예측
        predictions = []
        for i in range(self.num_samples):
            with torch.no_grad():
                # 모델에 noise 추가하여 불확실성 시뮬레이션
                noisy_features = input_features + torch.randn_like(input_features) * 0.01
                output = self.whisper.generate(
                    input_features=noisy_features,
                    max_length=448,
                    language=language,
                    task=task
                )
            predictions.append(self.processor.batch_decode(output, skip_special_tokens=True)[0])
            logger.debug(f"Sample {i+1}/{self.num_samples} completed")
        
        # 예측 결과 집계
        final_prediction = max(set(predictions), key=predictions.count)
        uncertainty = len(set(predictions)) / self.num_samples
        
        return final_prediction, uncertainty
    
    def transcribe_with_uncertainty(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Tuple[str, float]:
        """
        불확실성을 포함한 음성 인식 수행
        
        Args:
            audio_path (str): 오디오 파일 경로
            language (str, optional): 언어 코드 (기본값: 'ko' - 한국어)
            task (str): 작업 유형
            
        Returns:
            Tuple[str, float]: (인식 결과, 불확실성)
        """
        return self.transcribe(audio_path, language, task) 