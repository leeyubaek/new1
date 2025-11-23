from PyQt6.QtWidgets import QMainWindow, QTabWidget
from PyQt6.QtGui import QIcon
from pathlib import Path

# UI 탭 임포트
from app_ui.tabs.monitor_tab import MonitorTab
from app_ui.tabs.ranking_tab import RankingTab
from app_ui.tabs.profile_tab import ProfileTab
from app_ui.tabs.tutorial_tab import TutorialTab

# 스레드 및 데이터 관리 임포트
from app_ui.monitor_thread_mock import MonitorThread
import app_ui.data_manager as data_manager

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("자세 교정 도우미 (Posture Corrector)")
        self.setGeometry(100, 100, 900, 700)
        
        # (옵션) 아이콘 설정
        # icon_path = Path(__file__).parent / "icons" / "app_icon.png"
        # self.setWindowIcon(QIcon(str(icon_path)))

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # 모니터링 스레드 초기화
        self.monitor_thread = MonitorThread()
        self.current_session_data = {"total": 0, "correct": 0}
        
        # 탭 생성 및 추가
        self.init_tabs()
        
        # 시그널 연결
        self.connect_signals()

    def init_tabs(self):
        # 각 탭 위젯 생성
        self.monitor_tab = MonitorTab()
        self.ranking_tab = RankingTab()
        self.profile_tab = ProfileTab()
        self.tutorial_tab = TutorialTab()
        
        # QTabWidget에 탭 추가
        self.tabs.addTab(self.monitor_tab, "자세 측정")
        self.tabs.addTab(self.ranking_tab, "랭킹")
        self.tabs.addTab(self.profile_tab, "프로필")
        self.tabs.addTab(self.tutorial_tab, "도움말")

    def connect_signals(self):
        # 1. 모니터 탭의 버튼 -> 메인 윈도우의 스레드 제어 함수
        self.monitor_tab.start_button.clicked.connect(self.start_monitoring)
        self.monitor_tab.stop_button.clicked.connect(self.stop_monitoring)
        
        # 2. 스레드의 시그널 -> 모니터 탭의 UI 업데이트 슬롯
        self.monitor_thread.frame_ready.connect(self.monitor_tab.update_frame)
        self.monitor_thread.posture_status.connect(self.monitor_tab.update_posture_status)
        self.monitor_thread.timer_updated.connect(self.monitor_tab.update_timers)
        
        # 3. 스레드의 타이머 시그널 -> 메인 윈도우의 세션 데이터 업데이트
        self.monitor_thread.timer_updated.connect(self.update_session_data)

    def start_monitoring(self):
        print("메인: 모니터링 시작")
        self.monitor_tab.start_button.setEnabled(False)
        self.monitor_tab.stop_button.setEnabled(True)
        
        # 스레드 시작
        if not self.monitor_thread.isRunning():
            self.monitor_thread.start()
        
        self.monitor_tab.current_posture_label.setText("측정 중...")

    def stop_monitoring(self):
        print("메인: 모니터링 중지")
        if self.monitor_thread.isRunning():
            self.monitor_thread.stop()
        
        self.monitor_tab.start_button.setEnabled(True)
        self.monitor_tab.stop_button.setEnabled(False)
        
        # 모니터 탭 UI 초기화
        self.monitor_tab.reset_ui()

        # 현재 세션 기록 저장
        if self.current_session_data["total"] > 0:
            data_manager.add_ranking_entry(self.current_session_data)
            print("메인: 세션 저장 완료")
            # 랭킹 탭 새로고침
            self.ranking_tab.load_ranking()
        
        # 세션 데이터 초기화
        self.current_session_data = {"total": 0, "correct": 0}

    def update_session_data(self, total_sec, correct_sec):
        """스레드에서 받은 타이머 정보 저장"""
        self.current_session_data = {"total": total_sec, "correct": correct_sec}

    def closeEvent(self, event):
        """윈도우 종료 시 스레드 정리"""
        print("애플리케이션 종료 중...")
        if self.monitor_thread.isRunning():
            self.monitor_thread.stop()
            self.monitor_thread.wait() # 스레드가 완전히 끝날 때까지 대기
        event.accept()