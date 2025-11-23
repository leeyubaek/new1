from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFrame, QFormLayout
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor

# data_manager의 format_time 함수 임포트
from app_ui.data_manager import format_time

class MonitorTab(QWidget):
    """
    자세 측정 탭 UI
    """
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        
        # 좌측: 비디오 피드 및 컨트롤
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)
        
        self.video_label = QLabel("웹캠 피드를 여기에 표시합니다.")
        self.video_label.setObjectName("VideoLabel") # 스타일시트 적용용
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedSize(640, 480)
        left_layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("시작 (Start)")
        
        self.stop_button = QPushButton("종료 (Stop)")
        self.stop_button.setEnabled(False)
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        left_layout.addLayout(control_layout)
        
        main_layout.addLayout(left_layout, 2) # 좌측 영역이 2의 비율
        
        # 우측: 상태 및 정보
        right_layout = QVBoxLayout()
        right_layout.setSpacing(20)
        right_layout.setContentsMargins(10, 0, 10, 0)
        
        # 상태 프레임
        status_frame = QFrame()
        status_frame.setFrameShape(QFrame.Shape.StyledPanel)
        status_frame.setStyleSheet("background-color: #3A3A3A; border-radius: 5px;")
        status_layout = QVBoxLayout(status_frame)
        status_layout.setSpacing(10)
        
        status_title = QLabel("현재 상태")
        status_title.setObjectName("StatusTitle")
        status_layout.addWidget(status_title, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.current_posture_label = QLabel("대기 중...")
        self.current_posture_label.setObjectName("StatusLabel")
        self.current_posture_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.current_posture_label)
        
        right_layout.addWidget(status_frame)
        
        # 타이머 프레임
        timer_frame = QFrame()
        timer_frame.setFrameShape(QFrame.Shape.StyledPanel)
        timer_frame.setStyleSheet("background-color: #3A3A3A; border-radius: 5px;")
        timer_layout = QFormLayout(timer_frame)
        timer_layout.setSpacing(15)
        
        timer_title = QLabel("측정 시간")
        timer_title.setFont(QFont("Malgun Gothic", 14, QFont.Weight.Bold))
        timer_layout.addRow(timer_title)

        self.total_time_label = QLabel("00:00:00")
        self.total_time_label.setObjectName("TimerLabel")
        timer_layout.addRow("총 시간:", self.total_time_label)
        
        self.correct_time_label = QLabel("00:00:00")
        self.correct_time_label.setObjectName("CorrectTimerLabel")
        timer_layout.addRow("바른 자세 시간:", self.correct_time_label)

        right_layout.addWidget(timer_frame)
        
        # 환경 조언 (Placeholder)
        env_frame = QFrame()
        env_frame.setFrameShape(QFrame.Shape.StyledPanel)
        env_frame.setStyleSheet("background-color: #3A3A3A; border-radius: 5px;")
        env_layout = QVBoxLayout(env_frame)
        
        env_title = QLabel("업무 환경 조언 (준비 중)")
        env_title.setFont(QFont("Malgun Gothic", 14, QFont.Weight.Bold))
        env_layout.addWidget(env_title)
        
        self.env_advice_label = QLabel("모니터, 의자, 책상 높이 분석 기능이\n여기에 추가될 예정입니다.")
        self.env_advice_label.setStyleSheet("color: #888;")
        env_layout.addWidget(self.env_advice_label)
        
        right_layout.addWidget(env_frame)

        right_layout.addStretch(1) # 남은 공간 채우기
        
        main_layout.addLayout(right_layout, 1) # 우측 영역이 1의 비율

    # --- 스레드로부터 호출될 슬롯 ---
    
    def update_frame(self, frame):
        """웹캠 프레임을 QLabel에 표시"""
        try:
            # frame (np.ndarray)을 QPixmap으로 변환
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            # BGR -> RGB 변환
            rgb_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888).rgbSwapped()
            pixmap = QPixmap.fromImage(rgb_image)
            
            scaled_pixmap = pixmap.scaled(self.video_label.size(), 
                                          Qt.AspectRatioMode.KeepAspectRatio, 
                                          Qt.TransformationMode.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"프레임 업데이트 오류: {e}")

    def update_posture_status(self, status):
        """자세 상태 텍스트 및 색상 업데이트"""
        if status == "normal":
            self.current_posture_label.setText("바른 자세")
            self.current_posture_label.setStyleSheet("color: #4CAF50;") # 녹색
        elif status == "abnormal":
            self.current_posture_label.setText("나쁜 자세!")
            self.current_posture_label.setStyleSheet("color: #F44336;") # 빨간색
        else:
            self.current_posture_label.setText(status)
            self.current_posture_label.setStyleSheet("color: #E0E0E0;") # 기본색

    def update_timers(self, total_sec, correct_sec):
        """타이머 라벨 업데이트"""
        self.total_time_label.setText(format_time(total_sec))
        self.correct_time_label.setText(format_time(correct_sec))

    def reset_ui(self):
        """모니터링 종료 시 UI 초기화"""
        self.video_label.setText("웹캠 피드를 여기에 표시합니다.")
        self.video_label.setPixmap(QPixmap()) # 이미지 초기화
        self.current_posture_label.setText("대기 중...")
        self.current_posture_label.setStyleSheet("color: #E0E0E0;")
        self.total_time_label.setText("00:00:00")
        self.correct_time_label.setText("00:00:00")