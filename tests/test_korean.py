import sys
import logging
from pathlib import Path
import time
import gc

# src 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.bnn_whisper import BNNWhisper
from src.audio_recorder import AudioRecorder
from src.utils import setup_logging, log_memory_usage, clear_memory, get_memory_usage

def test_korean_speech_recognition():
    """한국어 음성 인식 테스트"""
    # 로깅 설정
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # CPU 모델 초기화
    logger.info("\n=== Testing with CPU ===")
    cpu_model = BNNWhisper(
        model_name="openai/whisper-base",
        device="cpu",
        num_samples=10,
        default_language="ko"
    )
    
    # MPS 모델 초기화
    logger.info("\n=== Testing with MPS ===")
    mps_model = BNNWhisper(
        model_name="openai/whisper-base",
        device="mps",
        num_samples=10,
        default_language="ko"
    )
    
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
                # CPU로 테스트
                logger.info("\n=== CPU Test ===")
                cpu_result, cpu_uncertainty, cpu_metrics = cpu_model.transcribe_with_uncertainty(
                    audio_file,
                    language="ko",
                    task="transcribe"
                )
                
                # MPS로 테스트
                logger.info("\n=== MPS Test ===")
                mps_result, mps_uncertainty, mps_metrics = mps_model.transcribe_with_uncertainty(
                    audio_file,
                    language="ko",
                    task="transcribe"
                )
                
                # 결과 출력
                print("\n=== 성능 비교 결과 ===")
                print(f"CPU 결과: {cpu_result}")
                print(f"CPU 불확실성: {cpu_uncertainty:.2f}")
                print(f"CPU 총 소요시간: {cpu_metrics.total_time:.2f}초")
                print(f"CPU 전처리 시간: {cpu_metrics.preprocessing_time:.2f}초")
                print(f"CPU 추론 시간: {cpu_metrics.inference_time:.2f}초")
                print(f"CPU 메모리 사용량: {cpu_metrics.memory_usage['rss']:.2f} MB")
                
                print(f"\nMPS 결과: {mps_result}")
                print(f"MPS 불확실성: {mps_uncertainty:.2f}")
                print(f"MPS 총 소요시간: {mps_metrics.total_time:.2f}초")
                print(f"MPS 전처리 시간: {mps_metrics.preprocessing_time:.2f}초")
                print(f"MPS 추론 시간: {mps_metrics.inference_time:.2f}초")
                print(f"MPS 메모리 사용량: {mps_metrics.memory_usage['rss']:.2f} MB")
                
                # 성능 향상률 계산
                speedup = cpu_metrics.total_time / mps_metrics.total_time
                print(f"\nMPS 성능 향상률: {speedup:.2f}x")
                print("=====================")
                
            finally:
                # 임시 파일 정리
                recorder.cleanup(audio_file)
                
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        raise
    finally:
        # 메모리 정리
        clear_memory()

if __name__ == "__main__":
    test_korean_speech_recognition() 