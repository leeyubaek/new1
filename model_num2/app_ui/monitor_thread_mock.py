"""
(Mock Version)
UI 테스트를 위한 가짜 모니터 스레드.
- tensorflow, mediapipe, cv2 등 무거운 라이브러리를 임포트하지 않습니다.
- DLL 오류를 우회하여 UI 테스트를 진행할 수 있습니다.
- 2초마다 "normal" / "abnormal" 상태를 랜덤으로 생성합니다.
"""
import numpy as np
import time
import random
from PyQt6.QtCore import QThread, pyqtSignal

class MonitorThread(QThread): # 클래스 이름은 원본과 동일하게 유지
    """
    (Mock) 웹캠 대신 가짜 프레임과 가짜 자세 데이터를 생성합니다.
    """
    frame_ready = pyqtSignal(np.ndarray)  
    posture_status = pyqtSignal(str)      
    timer_updated = pyqtSignal(int, int)  
    
    def __init__(self):
        super().__init__()
        self._is_running = False
        self.total_time_sec = 0
        self.correct_time_sec = 0
        self.start_time = None
        self.current_posture = "대기 중"
        self.last_status_change_time = 0

    def load_models(self):
        """(Mock) 모델 로드 건너뛰기"""
        print("[MOCK] 모델 로드를 건너뜁니다.")
        return True

    def run(self):
        if not self.load_models():
            self._is_running = False
            return

        self._is_running = True
        
        self.total_time_sec = 0
        self.correct_time_sec = 0
        self.start_time = time.time()
        last_timer_update = time.time()
        self.last_status_change_time = time.time()
        self.current_posture = "normal"

        print("[MOCK] 가짜 모니터링 스레드 시작...")

        while self._is_running:
            
            # --- [MOCK DATA] ---
            # 1. 가짜 비디오 프레임 생성 (검은 화면)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 2. 2초마다 가짜 자세 상태 변경
            current_time = time.time()
            if current_time - self.last_status_change_time > 2.0:
                self.current_posture = random.choice(["normal", "abnormal"])
                self.last_status_change_time = current_time
                print(f"[MOCK] 자세 변경 -> {self.current_posture}")
            
            # 3. 가짜 프레임에 텍스트 그리기 (OpenCV 없이)
            # (UI의 StatusLabel이 있으므로 이 부분은 생략 가능)
            
            # --- [MOCK EMIT] ---

            # GUI로 프레임 전송
            self.frame_ready.emit(frame)
            
            # GUI로 자세 상태 전송
            self.posture_status.emit(self.current_posture)
            
            # 타이머 업데이트 (1초마다)
            if current_time - last_timer_update >= 1.0:
                elapsed_since_start = current_time - self.start_time
                self.total_time_sec = int(elapsed_since_start)
                
                if self.current_posture == "normal":
                    self.correct_time_sec += 1 
                
                self.timer_updated.emit(self.total_time_sec, self.correct_time_sec)
                last_timer_update = current_time

            # CPU 사용량 조절 (약 30fps)
            time.sleep(0.03)

        print("[MOCK] 모니터링 스레드 종료됨")

    def stop(self):
        self._is_running = False