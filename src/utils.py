import os
import torch
import logging
import psutil
import gc
from typing import Optional, List
from memory_profiler import profile

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

def get_memory_usage():
    """현재 프로세스의 메모리 사용량을 반환"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
    }

def log_memory_usage(label: str = "Current"):
    """메모리 사용량을 로깅"""
    memory = get_memory_usage()
    logger.info(f"{label} Memory Usage - RSS: {memory['rss']:.2f} MB, VMS: {memory['vms']:.2f} MB")

def clear_memory():
    """메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Memory cleared")
