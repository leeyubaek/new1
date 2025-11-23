from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QTableWidget, 
    QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
import app_ui.data_manager as data_manager

class RankingTab(QWidget):
    """
    랭킹 탭 UI
    """
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        title = QLabel("나의 자세 기록 (랭킹)")
        title.setFont(QFont("Malgun Gothic", 16, QFont.Weight.Bold))
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.ranking_table = QTableWidget()
        self.ranking_table.setColumnCount(4)
        self.ranking_table.setHorizontalHeaderLabels(["날짜", "총 측정 시간", "바른 자세 시간", "바른 자세 비율"])
        self.ranking_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.ranking_table.horizontalHeader().setStretchLastSection(True)
        self.ranking_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers) # 수정 불가
        
        layout.addWidget(self.ranking_table)
        
        self.load_ranking_button = QPushButton("기록 새로고침")
        self.load_ranking_button.clicked.connect(self.load_ranking)
        layout.addWidget(self.load_ranking_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        self.load_ranking() # 탭 생성 시 기록 로드

    def load_ranking(self):
        """랭킹 데이터를 테이블에 표시"""
        records = data_manager.load_ranking()
        
        # 최신순으로 정렬
        records.sort(key=lambda x: x['date'], reverse=True)
        
        self.ranking_table.setRowCount(len(records))
        
        for i, record in enumerate(records):
            total_sec = record.get("total_time", 0)
            correct_sec = record.get("correct_time", 0)
            
            if total_sec > 0:
                percentage = f"{(correct_sec / total_sec) * 100:.1f}%"
            else:
                percentage = "N/A"
            
            self.ranking_table.setItem(i, 0, QTableWidgetItem(record.get("date", "")))
            self.ranking_table.setItem(i, 1, QTableWidgetItem(data_manager.format_time(total_sec)))
            self.ranking_table.setItem(i, 2, QTableWidgetItem(data_manager.format_time(correct_sec)))
            self.ranking_table.setItem(i, 3, QTableWidgetItem(percentage))
            
            # 비율에 따라 색상 적용 (옵션)
            try:
                percent_val = float(percentage[:-1])
                if percent_val < 50.0:
                    color = QColor("#F44336") # Red
                elif percent_val < 80.0:
                    color = QColor("#FFC107") # Amber
                else:
                    color = QColor("#4CAF50") # Green
                self.ranking_table.item(i, 3).setForeground(color)
            except ValueError:
                pass # N/A 등 예외 처리

        self.ranking_table.resizeRowsToContents()