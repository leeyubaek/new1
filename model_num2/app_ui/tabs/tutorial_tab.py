from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextBrowser

class TutorialTab(QWidget):
    """
    도움말 (튜토리얼) 탭 UI
    """
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.tutorial_browser = QTextBrowser()
        self.tutorial_browser.setOpenExternalLinks(True) # 링크 클릭 시 브라우저로 열기
        self.tutorial_browser.setHtml("""
            <h1 style='color: #E0E0E0;'>자세 교정 도우미 사용법</h1>
            <p style='font-size: 12pt;'>
                이 프로그램은 웹캠을 통해 사용자의 자세를 실시간으로 분석하고 피드백을 줍니다.
            </p>
            
            <h2 style='color: #4A90E2;'>1. 자세 측정 탭</h2>
            <ul>
                <li><strong>[시작]</strong> 버튼을 누르면 웹캠이 켜지고 자세 측정이 시작됩니다.</li>
                <li>프로그램이 사용자의 자세를 'normal'(바른 자세) 또는 'abnormal'(나쁜 자세)로 판단합니다.</li>
                <li>'총 시간'과 '바른 자세 시간'이 실시간으로 기록됩니다.</li>
                <li><strong>[종료]</strong> 버튼을 누르면 측정이 중지되고, '랭킹' 탭에 기록이 저장됩니다.</li>
            </ul>

            <h2 style='color: #4A90E2;'>2. 랭킹 탭</h2>
            <ul>
                <li>과거의 자세 측정 기록을 모두 확인할 수 있습니다.</li>
                <li>'바른 자세 비율'을 통해 자신의 자세 습관을 모니터링해 보세요.</li>
            </ul>

            <h2 style='color: #4A90E2;'>3. 프로필 탭</h2>
            <ul>
                <li>닉네임, 키, 몸무게 등 기본 정보를 입력하고 저장할 수 있습니다. (향후 분석에 사용될 수 있습니다)</li>
                <li>간단한 아바타를 선택할 수 있습니다.</li>
            </ul>
            
            <p style='color: #AAA; margin-top: 20px;'>
                <strong>참고:</strong> 정확한 측정을 위해 카메라 정면에 상반신이 잘 보이도록 앉아주세요.
                조명이 너무 어둡거나 밝으면 정확도가 떨어질 수 있습니다.
            </p>
        """)
        layout.addWidget(self.tutorial_browser)