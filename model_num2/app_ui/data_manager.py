import os
import json
from datetime import datetime

# 파일 경로는 main.py에서 os.chdir()로 설정한 루트 폴더 기준입니다.
DATA_DIR = "data"
PROFILE_FILE = os.path.join(DATA_DIR, "posture_profile.json")
RANKING_FILE = os.path.join(DATA_DIR, "posture_ranking.json")

def format_time(seconds):
    """초를 HH:MM:SS 형식의 문자열로 변환"""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:02}"

# --- 프로필 관리 ---

def load_profile():
    """프로필 파일에서 데이터 읽기"""
    if not os.path.exists(PROFILE_FILE):
        return {} # 파일이 없으면 빈 딕셔너리
    
    try:
        with open(PROFILE_FILE, 'r', encoding='utf-8') as f:
            profile_data = json.load(f)
        return profile_data
    except Exception as e:
        print(f"프로필 로드 오류: {e}")
        return {}

def save_profile(data):
    """프로필 데이터를 파일에 저장"""
    try:
        # data 폴더가 없으면 생성
        os.makedirs(DATA_DIR, exist_ok=True)
        
        with open(PROFILE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"프로필 저장 완료: {PROFILE_FILE}")
    except Exception as e:
        print(f"프로필 저장 오류: {e}")

# --- 랭킹 관리 ---

def load_ranking():
    """랭킹 파일에서 데이터 읽기"""
    if not os.path.exists(RANKING_FILE):
        return [] # 파일이 없으면 빈 리스트
    
    try:
        with open(RANKING_FILE, 'r', encoding='utf-8') as f:
            records = json.load(f)
        return records
    except Exception as e:
        print(f"랭킹 로드 오류: {e}")
        return []

def add_ranking_entry(session_data):
    """현재 세션의 타이머 데이터를 랭킹 파일에 추가"""
    if session_data["total"] == 0:
        return # 측정 시간이 없으면 저장 안 함

    new_record = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_time": session_data["total"],
        "correct_time": session_data["correct"]
    }
    
    records = load_ranking()
    records.append(new_record)
    
    try:
        # data 폴더가 없으면 생성
        os.makedirs(DATA_DIR, exist_ok=True)
        
        with open(RANKING_FILE, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=4)
        print(f"세션 기록 저장 완료: {RANKING_FILE}")
    except Exception as e:
        print(f"세션 기록 저장 오류: {e}")