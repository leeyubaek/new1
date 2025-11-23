"""
메인 애플리케이션 실행 파일 (Entry Point)
"""
import sys
import os
from PyQt6.QtWidgets import QApplication

# --- [중요] 경로 설정 ---
# 현재 파일(main.py)의 위치를 기준으로 루트 폴더(model_num1)의 경로를 계산합니다.
current_file_path = os.path.abspath(__file__)
app_ui_dir = os.path.dirname(current_file_path)
root_dir = os.path.dirname(app_ui_dir)

# 1. 파이썬 임포트 경로에 루트 폴더 추가
#    (src.preprocessing 등을 임포트하기 위해)
sys.path.append(root_dir)

# 2. 현재 작업 디렉토리를 루트 폴더로 변경
#    (models/..., data/... 등 상대 경로 파일 접근을 위해)
os.chdir(root_dir)

# --- 경로 설정 후 모듈 임포트 ---
from app_ui.main_window import MainWindow
from app_ui.style import STYLE_SHEET

def main():
    app = QApplication(sys.argv)
    
    # 스타일시트 적용
    app.setStyleSheet(STYLE_SHEET)
    
    # 메인 윈도우 생성 및 표시
    window = MainWindow()
    window.show()
    
    # 이벤트 루프 시작
    sys.exit(app.exec())

if __name__ == "__main__":
    main()