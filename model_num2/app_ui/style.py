"""
PyQt6 애플리케이션을 위한 QSS 스타일시트 (밝은 테마)
"""

STYLE_SHEET = """
/* --- 전역 설정 --- */
QWidget {
    background-color: #FFFFFF;       /* 기본 배경색 (흰색) */
    color: #333333;                 /* 기본 글자색 (어두운 회색) */
    font-family: 'Malgun Gothic', '맑은 고딕', sans-serif; /* 윈도우 기본 한글 글꼴 */
    font-size: 10pt;
}

QMainWindow {
    background-color: #F8F8F8;     /* 메인 윈도우 배경 (약간의 회색) */
}

/* --- 탭 위젯 --- */
QTabWidget::pane {
    border: 1px solid #E0E0E0;     /* 탭 내용 영역 테두리 */
    background-color: #FFFFFF;
}

QTabBar::tab {
    background-color: #F0F0F0;     /* 비활성 탭 배경 */
    color: #555555;               /* 비활성 탭 글자 */
    padding: 10px 20px;
    border: 1px solid #E0E0E0;
    border-bottom: none;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

QTabBar::tab:selected {
    background-color: #FFFFFF;     /* 활성 탭 배경 (흰색) */
    color: #000000;               /* 활성 탭 글자 (검은색) */
    border-bottom: 2px solid #0078D7; /* 활성 탭 하단 파란색 선 */
}

QTabBar::tab:hover {
    background-color: #FAFAFA;     /* 탭 호버 */
    color: #000000;
}

/* --- 버튼 --- */
QPushButton {
    background-color: #F0F0F0;     /* 기본 버튼 배경 */
    border: 1px solid #D0D0D0;     /* 기본 버튼 테두리 */
    padding: 8px 15px;
    border-radius: 4px;
    font-size: 10pt;
}
QPushButton:hover {
    background-color: #E6E6E6;     /* 버튼 호버 */
    border-color: #C0C0C0;
}
QPushButton:pressed {
    background-color: #DCDCDC;     /* 버튼 클릭 */
}
QPushButton:disabled {
    background-color: #F5F5F5;     /* 비활성 버튼 */
    color: #A0A0A0;
}

/* [시작] 버튼 (monitor_tab.py의 objectName) */
QPushButton#start_button {
    background-color: #4CAF50;     /* 녹색 */
    color: white;
    font-weight: bold;
}
QPushButton#start_button:hover { background-color: #45A049; }
QPushButton#start_button:pressed { background-color: #3E8E41; }

/* [종료] 버튼 (monitor_tab.py의 objectName) */
QPushButton#stop_button {
    background-color: #F44336;     /* 빨간색 */
    color: white;
    font-weight: bold;
}
QPushButton#stop_button:hover { background-color: #E53935; }
QPushButton#stop_button:pressed { background-color: #D32F2F; }

/* [저장] 버튼 (profile_tab.py의 objectName) */
QPushButton#profile_save_button {
    background-color: #0078D7;     /* 파란색 */
    color: white;
    font-weight: bold;
}
QPushButton#profile_save_button:hover { background-color: #005A9E; }
QPushButton#profile_save_button:pressed { background-color: #004C8A; }


/* --- 라벨 (QLabel) --- */

/* 비디오 피드 라벨 (monitor_tab.py의 objectName) */
QLabel#video_label {
    background-color: #000000;     /* 카메라 피드 배경 (검은색) */
    border: 1px solid #E0E0E0;
    border-radius: 4px;
}

/* 자세 상태 라벨 (monitor_tab.py의 objectName) */
QLabel#status_label_normal {
    color: #4CAF50;                /* 바른 자세 (녹색) */
    font-size: 16pt;
    font-weight: bold;
}
QLabel#status_label_abnormal {
    color: #F44336;                /* 나쁜 자세 (빨간색) */
    font-size: 16pt;
    font-weight: bold;
}

/* 타이머 라벨 (monitor_tab.py의 class) */
QLabel[class="timer_label"] {
    font-size: 20pt;
    font-weight: bold;
    color: #0078D7;                /* 타이머 (파란색) */
}
QLabel[class="timer_sublabel"] {
    font-size: 11pt;
    color: #555555;
}

/* --- 입력창 (QLineEdit, QSpinBox) --- */
QLineEdit, QSpinBox {
    padding: 8px;
    border: 1px solid #D0D0D0;
    border-radius: 4px;
    background-color: #FFFFFF;
}
QLineEdit:focus, QSpinBox:focus {
    border: 1.5px solid #0078D7;   /* 포커스 시 파란색 테두리 */
}

/* --- 테이블 (ranking_tab.py) --- */
QTableWidget {
    border: 1px solid #E0E0E0;
    gridline-color: #E8E8E8;       /* 테이블 그리드 라인 */
}
QHeaderView::section {
    background-color: #F8F8F8;     /* 테이블 헤더 배경 */
    padding: 8px;
    border: 1px solid #E0E0E0;
    font-weight: bold;
}
QTableWidget::item {
    padding: 5px;
}
QTableWidget::item:selected {
    background-color: #E0EAF6;     /* 테이블 선택 시 배경 */
    color: #333333;
}
"""