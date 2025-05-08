import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class AudioRecorder:
    def __init__(self, sample_rate: int = 16000):
        """
        오디오 녹음기 초기화
        
        Args:
            sample_rate (int): 샘플링 레이트 (기본값: 16000Hz)
        """
        self.sample_rate = sample_rate
        self.recording = False
        self.audio_data = []
        
    def _audio_callback(self, indata, frames, time, status):
        """오디오 데이터 콜백 함수"""
        if status:
            logger.warning(f"오디오 스트림 상태: {status}")
        if self.recording:
            self.audio_data.append(indata.copy())
    
    def start_recording(self):
        """녹음 시작"""
        self.recording = True
        self.audio_data = []
        logger.info("녹음을 시작합니다...")
        
        # 오디오 스트림 시작
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            callback=self._audio_callback
        )
        self.stream.start()
    
    def stop_recording(self) -> str:
        """
        녹음 중지 및 임시 파일 저장
        
        Returns:
            str: 저장된 오디오 파일 경로
        """
        self.recording = False
        self.stream.stop()
        self.stream.close()
        
        # 녹음된 데이터 처리
        audio_data = np.concatenate(self.audio_data, axis=0)
        
        # 임시 파일로 저장
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        wav.write(temp_file.name, self.sample_rate, audio_data)
        
        logger.info(f"녹음이 완료되었습니다. 파일 저장: {temp_file.name}")
        return temp_file.name
    
    def cleanup(self, file_path: str):
        """임시 파일 삭제"""
        try:
            os.unlink(file_path)
            logger.debug(f"임시 파일 삭제됨: {file_path}")
        except Exception as e:
            logger.warning(f"임시 파일 삭제 실패: {e}") 