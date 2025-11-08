import sounddevice as sd
import numpy as np
import time
import threading
import queue
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import lightning_whisper_mlx
from light_whisper_streaming import OnlineSTTProcessor
from light_whisper_streaming import KoreanTokenizer


class STTTester:
    """음성 인식 테스트 애플리케이션"""
    
    def __init__(self, model_size="small"):
        self.model_size = model_size
        self.rate = 16000
        self.channels = 1
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_running = False
        
        # 모델 로드
        print(f"Lightning Whisper MLX 모델 로딩 ({model_size})...")
        self.model = lightning_whisper_mlx.LightningWhisperMLX(model=model_size, batch_size=4)
        print(f"모델 로드 완료")
        
        # 처리기 생성
        self.processor = OnlineSTTProcessor(self.model)
    
    def audio_callback(self, indata, frames, time_info, status):
        """오디오 콜백 함수"""
        if status:
            print(f"오디오 상태: {status}")
        
        # 오디오 데이터 저장
        audio_data = indata[:, 0]
        self.audio_queue.put((audio_data.copy(), time.time()))
    
    def start(self):
        """테스트 시작"""
        if self.is_running:
            return
            
        self.is_running = True
        self.processor.init()
        
        # 마이크 스트림 시작
        self.stream = sd.InputStream(
            samplerate=self.rate,
            channels=self.channels,
            callback=self.audio_callback,
            dtype=np.float32
        )
        self.stream.start()
        
        # 처리 스레드 시작
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        print("마이크 녹음 시작. 종료하려면 Ctrl+C를 누르세요.")
    
    def process_audio(self):
        """오디오 처리 스레드"""
        last_process_time = time.time()
        min_chunk_size = 0.5  # 최소 청크 크기 (초)
        
        while self.is_running:
            try:
                # 최소 청크 크기만큼 수집
                audio_chunks = []
                chunk_samples = 0
                target_samples = int(min_chunk_size * self.rate)
                
                while chunk_samples < target_samples:
                    try:
                        audio_data, timestamp = self.audio_queue.get(timeout=0.1)
                        audio_chunks.append(audio_data)
                        chunk_samples += len(audio_data)
                        
                        # 너무 오래 기다리지 않도록 시간 체크
                        if time.time() - last_process_time > min_chunk_size * 2:
                            break
                    except queue.Empty:
                        if chunk_samples > 0:
                            break
                
                if not audio_chunks:
                    continue
                
                # 모든 청크 연결
                audio_data = np.concatenate(audio_chunks)
                
                # 처리기에 오디오 추가
                self.processor.insert_audio_chunk(audio_data)
                
                # 일정 간격으로 처리
                current_time = time.time()
                if current_time - last_process_time >= min_chunk_size:
                    _, _, text, info = self.processor.process_iter()
                    if text:
                        print(f"인식 중: {text}")
                    last_process_time = current_time
                    
            except Exception as e:
                print(f"오류: {str(e)}")
    
    def stop(self):
        """테스트 중지"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # 스트림 중지
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        # 최종 결과 처리
        _, _, final_text, _ = self.processor.finish()
        
        # print("\n=== 최종 인식 결과 ===")
        # if final_text:
        #     print(final_text)
        # else:
        #     print("인식된 텍스트 없음")
        # print("======================")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Lightning Whisper MLX 테스트')
    parser.add_argument('--model', default='small', choices=['tiny', 'base', 'small', 'medium', 'large-v3-turbo'],
                       help='모델 크기 (default: small)')
    
    args = parser.parse_args()
    
    tester = STTTester(model_size=args.model)

    try:
        tester.start()
        while tester.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n종료합니다...")
    finally:
        tester.stop()


if __name__ == "__main__":
    main()