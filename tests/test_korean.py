import sys
import logging
from pathlib import Path
import time
import gc

# src 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.bnn_whisper import BNNWhisper
from src.audio_recorder import AudioRecorder
from src.utils import setup_logging, log_memory_usage, clear_memory

def test_korean_speech_recognition():
    """한국어 음성 인식 테스트"""
    # 로깅 설정
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 초기 메모리 사용량 측정
    log_memory_usage("Initial")
    
    # 모델 초기화
    logger.info("Initializing model...")
    model = BNNWhisper(
        model_name="openai/whisper-base",
        num_samples=10,
        default_language="ko"
    )
    
    # 모델 로드 후 메모리 사용량 측정
    log_memory_usage("After model load")
    
    # 오디오 녹음기 초기화
    recorder = AudioRecorder()
    
    try:
        while True:
            input("\n녹음을 시작하려면 Enter를 누르세요... (종료하려면 Ctrl+C)")
            
            # 녹음 시작
            recorder.start_recording()
            print("녹음 중... (중지하려면 Enter를 누르세요)")
            input()
            
            # 녹음 중지 및 파일 저장
            audio_file = recorder.stop_recording()
            
            try:
                # 인식 전 메모리 사용량 측정
                log_memory_usage("Before transcription")
                
                # 음성 인식 수행
                result, uncertainty = model.transcribe_with_uncertainty(
                    audio_file,
                    language="ko",
                    task="transcribe"
                )
                
                # 인식 후 메모리 사용량 측정
                log_memory_usage("After transcription")
                
                print("\n=== 인식 결과 ===")
                print(f"텍스트: {result}")
                print(f"불확실성: {uncertainty:.2f}")
                print("===============")
                
                # 메모리 정리
                clear_memory()
                log_memory_usage("After cleanup")
                
            finally:
                # 임시 파일 정리
                recorder.cleanup(audio_file)
                
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        raise
    finally:
        # 최종 메모리 사용량 측정
        log_memory_usage("Final")
        clear_memory()

if __name__ == "__main__":
    test_korean_speech_recognition() 