import cv2
import numpy as np
import tensorflow as tf
import joblib
import json
import time
import sys
import os # os.path.join을 위해 임포트
from PyQt6.QtCore import QThread, pyqtSignal

# --- [중요] ---
# main.py에서 sys.path.append(루트폴더)를 실행했기 때문에
# 루트/src/preprocessing.py 임포트가 가능합니다.
try:
    from src.preprocessing import PosePreprocessor
except ImportError as e:
    print(f"오류: src.preprocessing를 임포트할 수 없습니다. (에러: {e})")
    print("main.py가 올바른 sys.path를 설정했는지 확인하세요.")
    sys.exit(1)


# --- 모델 및 파일 경로 ---
# main.py에서 os.chdir(루트폴더)를 실행했기 때문에
# 루트 폴더 기준으로 상대 경로를 작성합니다.
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, 'cnn_lstm_model.h5')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')
METADATA_PATH = os.path.join(MODELS_DIR, 'model_metadata.json')


class MonitorThread(QThread):
    """
    웹캠 피드 처리 및 자세 추론을 담당하는 QThread
    GUI가 멈추는 것을 방지합니다.
    """
    # GUI로 보낼 시그널 정의
    frame_ready = pyqtSignal(np.ndarray)  # 비디오 프레임
    posture_status = pyqtSignal(str)      # 자세 상태 (e.g., "normal", "abnormal")
    timer_updated = pyqtSignal(int, int)  # (총 시간, 바른 자세 시간)
    
    def __init__(self):
        super().__init__()
        self._is_running = False
        self.cap = None
        
        self.total_time_sec = 0
        self.correct_time_sec = 0
        self.start_time = None
        
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.metadata = None
        self.preprocessor = None
        
        self.sequence_data = []
        self.sequence_length = 10 # 기본값, 메타데이터에서 덮어씀
        self.feature_columns = []

    def load_models(self):
        """딥러닝 모델과 전처리기 로드"""
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.label_encoder = joblib.load(ENCODER_PATH)
            
            with open(METADATA_PATH, 'r') as f:
                self.metadata = json.load(f)
            
            self.sequence_length = self.metadata['sequence_length']
            self.feature_columns = self.metadata['feature_columns']
            
            # MediaPipe Pose 초기화를 포함한 전처리기 (src.preprocessing)
            self.preprocessor = PosePreprocessor()
            
            print("모델 및 전처리기 로드 완료")
            return True
        except Exception as e:
            print(f"모델 로드 오류: {e}")
            print(f"필요한 파일({MODEL_PATH}, {SCALER_PATH}, {ENCODER_PATH}, {METADATA_PATH})이 {MODELS_DIR} 폴더에 있는지 확인하세요.")
            return False

    def run(self):
        if not self.load_models():
            self._is_running = False
            return # 모델 로드 실패 시 스레드 종료

        self._is_running = True
        self.cap = cv2.VideoCapture(0) # 0번 카메라
        
        if not self.cap.isOpened():
            print("오류: 카메라를 열 수 없습니다.")
            self._is_running = False
            return

        self.total_time_sec = 0
        self.correct_time_sec = 0
        self.sequence_data = []
        self.start_time = time.time()
        last_timer_update = time.time()

        current_posture = "대기 중"

        while self._is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1) # 좌우 반전

            # --- 실제 모델 추론 ---
            try:
                # 1. MediaPipe로 랜드마크 추출
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.preprocessor.pose.process(frame_rgb)

                feature_vector = None
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # 2. 특징(각도, 좌표) 추출 (src.preprocessing.py의 로직 사용)
                    angles_dict = self.preprocessor.calculate_angles(landmarks)
                    
                    feature_vector = []
                    # model_metadata.json에 정의된 순서대로 특징 추출
                    for col in self.feature_columns:
                        if col in angles_dict:
                            feature_vector.append(angles_dict[col])
                        else:
                            # (x, y 좌표 처리 - metadata에 정의된 랜드마크 기준)
                            try:
                                # UPPER_BODY_LANDMARKS는 preprocessing.py의 전역 변수
                                # 이 클래스에서 직접 접근하려면 preprocessor 객체를 통해 접근해야 하나,
                                # 현재 preprocessing.py의 UPPER_BODY_LANDMARKS는 전역 변수이므로
                                # PosePreprocessor 클래스에 해당 변수를 멤버로 추가하는 것이 좋습니다.
                                # (임시로 하드코딩된 이름 사용 가정)
                                
                                # preprocessing.py의 UPPER_BODY_LANDMARKS 딕셔너리 참조
                                landmark_map = {
                                    'nose': 0,
                                    'left_shoulder': 11,
                                    'right_shoulder': 12,
                                    'left_hip': 23,
                                    'right_hip': 24
                                }
                                
                                lm_name, axis = col.rsplit('_', 1) # e.g., "nose_x" -> ("nose", "x")
                                lm_index = landmark_map[lm_name]
                                lm = landmarks[lm_index]
                                feature_vector.append(getattr(lm, axis))
                            except (KeyError, AttributeError, ValueError):
                                # print(f"Warning: Cannot find feature {col}")
                                feature_vector.append(0) # 오류 시 0으로 대체

                if feature_vector and len(feature_vector) == len(self.feature_columns):
                    # 3. 시퀀스 데이터 구성
                    self.sequence_data.append(feature_vector)
                    
                    if len(self.sequence_data) == self.sequence_length:
                        # 4. 스케일링 및 예측
                        scaled_data = self.scaler.transform(self.sequence_data)
                        prediction = self.model.predict(np.expand_dims(scaled_data, axis=0))
                        label_idx = np.argmax(prediction)
                        current_posture = self.label_encoder.classes_[label_idx]
                        
                        # 5. 시퀀스 데이터 슬라이딩 (가장 오래된 데이터 제거)
                        self.sequence_data.pop(0)

            except Exception as e:
                # print(f"추론 중 오류: {e}") # 디버깅용
                pass # 오류가 발생해도 계속 실행

            # --- 추론 종료 ---

            # 프레임에 현재 상태 표시
            display_text = f"Status: {current_posture}"
            color = (0, 255, 0) if current_posture == "normal" else (0, 0, 255)
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # GUI로 프레임 전송 (BGR 포맷 그대로)
            self.frame_ready.emit(frame)
            
            # GUI로 자세 상태 전송
            self.posture_status.emit(current_posture)
            
            # 타이머 업데이트 (1초마다)
            current_time = time.time()
            if current_time - last_timer_update >= 1.0:
                elapsed_since_start = current_time - self.start_time
                self.total_time_sec = int(elapsed_since_start)
                
                if current_posture == "normal":
                    self.correct_time_sec += 1 
                
                self.timer_updated.emit(self.total_time_sec, self.correct_time_sec)
                last_timer_update = current_time

            # CPU 사용량 조절 (약 30fps)
            time.sleep(0.03)

        # 스레드 종료 시 카메라 해제
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # MediaPipe 리소스 해제
        if self.preprocessor and hasattr(self.preprocessor, 'pose'):
            self.preprocessor.pose.close()
            
        print("모니터링 스레드 종료됨")

    def stop(self):
        self._is_running = False