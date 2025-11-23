# 🤖 웹캠 기반 실시간 자세 판단 AI 시스템 - 실행 가이드

> MediaPipe와 LSTM을 활용한 실시간 자세 교정 AI 시스템의 완전한 실행 가이드입니다.

## 📋 목차
1. [시스템 개요](#시스템-개요)
2. [환경 설정](#환경-설정)
3. [빠른 시작](#빠른-시작)
4. [단계별 실행](#단계별-실행)
5. [고급 사용법](#고급-사용법)
6. [문제 해결](#문제-해결)
7. [성능 최적화](#성능-최적화)

## 📊 시스템 개요

### 주요 기능
- **실시간 웹캠 자세 분석**: MediaPipe를 이용한 실시간 상체 스켈레톤 검출 및 분석
- **AI 기반 자세 판단**: LSTM 신경망을 통한 정상/비정상 자세 분류
- **시각적 피드백**: 실시간 자세 상태 표시 및 교정 알림

### 시스템 구성
```
📁 프로젝트 구조
├── 📂 data/                    # 데이터 저장소
│   ├── 📂 train_images/       # 훈련용 이미지
│   │   ├── 📂 normal/         # 정상 자세 이미지
│   │   └── 📂 abnormal/       # 비정상 자세 이미지
│   └── 📄 pose_data.csv       # 처리된 포즈 데이터
├── 📂 models/                  # 학습된 모델
│   ├── 📄 posture_lstm_model.h5  # LSTM 모델
│   ├── 📄 scaler.pkl          # 데이터 스케일러
│   └── 📄 label_encoder.pkl   # 라벨 인코더
├── 📂 src/                     # 소스 코드
│   ├── 📄 preprocessing.py    # 1단계: 이미지 전처리
│   ├── 📄 lstm_model.py      # 2단계: LSTM 모델 훈련
│   └── 📄 realtime_cam.py    # 3단계: 실시간 웹캠 시스템
├── 📄 run_system.py           # 통합 실행 스크립트
├── 📄 start.bat              # Windows 배치 실행 파일
└── 📄 requirements.txt       # 의존성 패키지
```

## 🔧 환경 설정

### 필수 요구사항
- **운영체제**: Windows 10/11
- **Python**: 3.8 - 3.11 (권장: 3.10)
- **웹캠**: USB 카메라 또는 내장 카메라
- **메모리**: 최소 8GB RAM (권장: 16GB)

### 1. Python 환경 설정

```powershell
# 1. 프로젝트 폴더로 이동
cd "C:\Users\user\OneDrive\Desktop\test\model_num1"

# 2. 가상환경 생성 (최초 1회만)
python -m venv .venv

# 3. 가상환경 활성화
.\.venv\Scripts\Activate.ps1

# 4. 패키지 업그레이드
python -m pip install --upgrade pip
```

### 2. 의존성 설치

```powershell
# 필수 패키지 설치
pip install -r requirements.txt
```

**설치되는 주요 패키지:**
- `mediapipe==0.10.21` - 실시간 포즈 추정
- `opencv-python==4.11.0.86` - 컴퓨터 비전
- `tensorflow==2.18.0` - 딥러닝 모델
- `pandas==2.3.3` - 데이터 처리
- `numpy==1.26.4` - 수치 연산
- `scikit-learn==1.6.0` - 머신러닝 도구
- `matplotlib==3.10.7` - 시각화

### 3. GPU 가속 설정 (선택사항)

```powershell
# CUDA가 설치된 경우 (성능 향상)
pip install tensorflow-gpu
```

## 🚀 빠른 시작

### 방법 1: 배치 파일 실행 (권장)

```powershell
# start.bat 더블클릭 또는 명령어로 실행
.\start.bat
```

**배치 파일 메뉴 구성:**
- `[1]` 📂 데이터 폴더 생성
- `[2]` 🚀 전체 시스템 실행 (전처리 → 훈련 → 모니터링)
- `[3]` 📊 1단계: 이미지 전처리만
- `[4]` 🧠 2단계: 모델 훈련만
- `[5]` 📹 3단계: 실시간 모니터링만

### 방법 2: 통합 스크립트 실행

```powershell
# 전체 시스템 한 번에 실행
.\.venv\Scripts\python.exe run_system.py --step all

# 샘플 데이터 폴더 생성
.\.venv\Scripts\python.exe run_system.py --setup
```

## 📊 단계별 실행

### 1️⃣ 1단계: 데이터 준비 및 전처리

#### 데이터 폴더 구조 생성
```powershell
# 샘플 데이터 폴더 생성
.\.venv\Scripts\python.exe run_system.py --setup
```

#### 훈련 이미지 준비
```
📂 data/train_images/
├── 📂 normal/          # 정상 자세 이미지 (최소 20장 이상)
│   ├── 🖼️ good1.jpg
│   ├── 🖼️ good2.jpg
│   └── ...
└── 📂 abnormal/        # 비정상 자세 이미지 (최소 20장 이상)
    ├── 🖼️ bad1.jpg
    ├── 🖼️ bad2.jpg
    └── ...
```

**이미지 품질 가이드:**
- ✅ 전신이 보이는 정면 이미지
- ✅ 선명하고 밝은 조명
- ✅ 다양한 각도와 배경
- ❌ 흐릿하거나 어두운 이미지
- ❌ 일부만 보이는 이미지

#### 전처리 실행
```powershell
# 자동 전처리
.\.venv\Scripts\python.exe run_system.py --step preprocess

# 또는 직접 실행
.\.venv\Scripts\python.exe src/preprocessing.py
```

**출력 결과:**
- `data/pose_data.csv` - 추출된 포즈 특성 데이터
- 로그에서 처리된 이미지 수 확인

### 2️⃣ 2단계: LSTM 모델 훈련

```powershell
# 자동 훈련
.\.venv\Scripts\python.exe run_system.py --step train

# 또는 직접 실행 (고급 옵션)
.\.venv\Scripts\python.exe src/lstm_model.py --data data/pose_data.csv --epochs 100
```

**훈련 과정:**
1. 📊 데이터 로드 및 전처리
2. 🧠 LSTM 신경망 구성
3. 🎯 모델 훈련 (50-100 에폭)
4. 📈 성능 평가 및 시각화
5. 💾 모델 저장

**출력 파일:**
- `models/posture_lstm_model.h5` - 훈련된 LSTM 모델
- `models/scaler.pkl` - 데이터 스케일러
- `models/label_encoder.pkl` - 라벨 인코더
- `models/confusion_matrix.png` - 혼동 행렬

### 3️⃣ 3단계: 실시간 웹캠 모니터링

```powershell
# 자동 실행
.\.venv\Scripts\python.exe run_system.py --step monitor

# 또는 직접 실행
.\.venv\Scripts\python.exe src/realtime_cam.py --camera 0
```

**실행 화면:**
- 🎥 실시간 웹캠 영상
- 🦴 실시간 스켈레톤 표시
- 📊 자세 예측 결과 (Normal/Abnormal)
- 📈 예측 확신도 (0-100%)

**조작법:**
- `ESC` 키: 프로그램 종료
- `R` 키: 시스템 리셋
- `S` 키: 스크린샷 저장

## 🔧 고급 사용법

### 개별 스크립트 실행

#### 전처리 고급 옵션
```powershell
# 라벨링된 데이터 처리
.\.venv\Scripts\python.exe src/preprocessing.py --input data/train_images --labeled --output data

# 단일 폴더 처리
.\.venv\Scripts\python.exe src/preprocessing.py --input single_folder --output data
```

#### LSTM 모델 고급 옵션
```powershell
# 하이퍼파라미터 조정
.\.venv\Scripts\python.exe src/lstm_model.py --data data/pose_data.csv --epochs 200 --batch_size 64 --sequence_length 10

# 빠른 테스트 (10 에폭)
.\.venv\Scripts\python.exe src/lstm_model.py --data data/pose_data.csv --epochs 10
```

#### 실시간 모니터링 고급 옵션
```powershell
# 특정 카메라 사용
.\.venv\Scripts\python.exe src/realtime_cam.py --camera 1

# 모델 경로 지정
.\.venv\Scripts\python.exe src/realtime_cam.py --model models/custom_model.h5
```

### 모델 파라미터 튜닝

**LSTM 모델 설정 (lstm_model.py):**
```python
# 기본 설정
sequence_length = 5      # 시퀀스 길이 (시간 단계)
lstm_units = 64         # LSTM 유닛 수
dropout_rate = 0.3      # 드롭아웃 비율
epochs = 50             # 훈련 에폭
batch_size = 16         # 배치 크기
```

**성능 향상을 위한 조정:**
- 데이터가 많은 경우: `epochs=100`, `batch_size=32`
- 데이터가 적은 경우: `epochs=30`, `dropout_rate=0.5`

## 🔍 문제 해결

### 일반적인 문제

#### 1. 가상환경 활성화 오류
```powershell
# PowerShell 실행 정책 설정
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 다시 활성화 시도
.\.venv\Scripts\Activate.ps1
```

#### 2. 카메라 접근 오류
- 다른 프로그램에서 카메라 사용 중인지 확인
- Windows 카메라 앱 종료
- 카메라 ID 변경: `--camera 1` 시도

#### 3. 모델 로드 실패
```powershell
# 모델 파일 존재 확인
dir models\

# 모델 재훈련
.\.venv\Scripts\python.exe run_system.py --step train
```

#### 4. 메모리 부족 오류
- 배치 크기 줄이기: `--batch_size 8`
- 이미지 해상도 낮추기
- 시퀀스 길이 줄이기: `--sequence_length 3`

#### 5. MediaPipe 설치 오류
```powershell
# Visual C++ 재배포 패키지 설치 후
pip uninstall mediapipe
pip install mediapipe==0.10.21
```

### 성능 문제

#### 낮은 정확도 해결
1. **더 많은 데이터 수집**: 각 클래스당 50장 이상
2. **데이터 품질 개선**: 선명하고 다양한 각도의 이미지
3. **하이퍼파라미터 조정**: 에폭 수 증가, 학습률 조정
4. **데이터 증강**: 노이즈 추가, 회전, 크기 조정

#### 실시간 처리 속도 개선
1. **GPU 사용**: CUDA 설치 및 tensorflow-gpu 사용
2. **해상도 낮추기**: 웹캠 해상도 640x480으로 설정
3. **모델 경량화**: LSTM 유닛 수 줄이기

### 디버깅 모드

```powershell
# 상세 로그 출력
.\.venv\Scripts\python.exe run_system.py --step all --verbose

# 개별 단계 디버깅
.\.venv\Scripts\python.exe src/preprocessing.py --debug
```

## 📈 성능 최적화

### 하드웨어 최적화

#### GPU 가속 설정
```powershell
# CUDA 11.8 설치 후
pip install tensorflow-gpu==2.18.0

# GPU 사용 확인
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

#### 메모리 최적화
```python
# GPU 메모리 점진적 할당 (코드에 추가)
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### 데이터 최적화

#### 이미지 전처리 최적화
- 해상도: 640x480 (실시간 처리용)
- 형식: JPG (파일 크기 최적화)
- 품질: 높은 대비와 선명도

#### 데이터 증강 기법
```python
# 자동 적용되는 증강 기법들
- 가우시안 노이즈 추가
- 좌표 스케일링 (±5%)
- 시계열 시프팅
```

### 모델 최적화

#### 경량화 설정
```python
# 빠른 추론을 위한 설정
sequence_length = 3      # 기본값: 5
lstm_units = 32         # 기본값: 64
```

#### 배치 처리 최적화
```python
# 실시간 처리용
batch_size = 1          # 실시간 단일 예측
```

## 📚 추가 정보

### 프로젝트 확장

#### 새로운 자세 클래스 추가
1. 새 폴더 생성: `data/train_images/new_posture/`
2. 이미지 추가 및 전처리 실행
3. 모델 재훈련

#### 웹 인터페이스 개발
```python
# Flask/FastAPI를 이용한 웹 서비스 구현
# REST API 엔드포인트 제공
```

#### 모바일 앱 연동
```python
# TensorFlow Lite 모델 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

### 관련 문서

- **README.md**: 프로젝트 개요 및 빠른 사용법
- **requirements.txt**: 의존성 패키지 목록
- **models/model_metadata.json**: 모델 설정 정보

### 기술 지원

문제 발생 시 다음 정보를 함께 제공해주세요:
1. 운영체제 및 Python 버전
2. 오류 메시지 전문
3. 실행 단계 및 명령어
4. 데이터 크기 및 형태

---

## 🎉 완료!

모든 단계를 성공적으로 완료하면 실시간 자세 교정 AI 시스템이 작동합니다!

**최종 확인 사항:**
- ✅ 가상환경 활성화
- ✅ 의존성 패키지 설치
- ✅ 훈련 데이터 준비 (각 클래스 20개 이상)
- ✅ 모델 훈련 완료 (정확도 80% 이상)
- ✅ 웹캠 실시간 모니터링 작동

**자세 교정 생활을 시작하세요! 🏃‍♀️💪**
