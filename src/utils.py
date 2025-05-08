import os
import torch
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO"):
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_available_devices() -> List[str]:
    """사용 가능한 디바이스 목록 반환"""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices

def ensure_directory(directory: str):
    """디렉토리가 존재하지 않으면 생성"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def validate_audio_file(file_path: str) -> bool:
    """오디오 파일 유효성 검사"""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    valid_extensions = ['.wav', '.mp3', '.flac']
    if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
        logger.error(f"Invalid file format. Supported formats: {', '.join(valid_extensions)}")
        return False
    
    return True
