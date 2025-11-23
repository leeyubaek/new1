from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, 
    QComboBox, QFormLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import app_ui.data_manager as data_manager

class ProfileTab(QWidget):
    """
    프로필 탭 UI
    """
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(20)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("사용자 프로필 설정")
        title.setFont(QFont("Malgun Gothic", 16, QFont.Weight.Bold))
        main_layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)

        form_layout = QFormLayout()
        form_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form_layout.setHorizontalSpacing(15)
        form_layout.setVerticalSpacing(15)

        self.nickname_input = QLineEdit("사용자")
        self.nickname_input.setMaxLength(20)
        form_layout.addRow("닉네임:", self.nickname_input)
        
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText("cm")
        form_layout.addRow("키:", self.height_input)
        
        self.weight_input = QLineEdit()
        self.weight_input.setPlaceholderText("kg")
        form_layout.addRow("몸무게:", self.weight_input)
        
        # 사용자 캐릭터 설정 (단순화)
        self.avatar_combo = QComboBox()
        self.avatar_combo.addItems(["기본 아바타 (A)", "아바타 B", "아바타 C", "표시 안함"])
        form_layout.addRow("캐릭터 설정:", self.avatar_combo)
        
        main_layout.addLayout(form_layout)
        
        self.save_profile_button = QPushButton("프로필 저장")
        self.save_profile_button.clicked.connect(self.save_profile)
        main_layout.addWidget(self.save_profile_button, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.status_label = QLabel("") # 저장 상태 표시
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

        self.load_profile() # 탭 생성 시 프로필 로드

    def load_profile(self):
        """프로필 데이터를 UI에 로드"""
        profile_data = data_manager.load_profile()
        
        self.nickname_input.setText(profile_data.get("nickname", "사용자"))
        self.height_input.setText(profile_data.get("height", ""))
        self.weight_input.setText(profile_data.get("weight", ""))
        self.avatar_combo.setCurrentText(profile_data.get("avatar", "기본 아바타 (A)"))
        print("프로필 로드 완료")

    def save_profile(self):
        """UI의 데이터를 프로필 파일에 저장"""
        profile_data = {
            "nickname": self.nickname_input.text(),
            "height": self.height_input.text(),
            "weight": self.weight_input.text(),
            "avatar": self.avatar_combo.currentText()
        }
        
        data_manager.save_profile(profile_data)
        self.status_label.setText("프로필이 저장되었습니다.")
        self.status_label.setStyleSheet("color: #4CAF50;")