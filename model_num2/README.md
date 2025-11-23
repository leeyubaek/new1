# 🤖 웹캠 기반 실시간 자세 판단 AI 시스템

MediaPipe와 CNN+LSTM 하이브리드 딥러닝 모델을 활용한 실시간 자세 교정 AI 시스템입니다.

## 📁 프로젝트 구조

```
model_num1/
├── data/                       # 데이터 폴더
│   ├── train_images/          # 훈련 이미지
│   │   ├── normal/            # 정상 자세 이미지
│   │   └── abnormal/          # 비정상 자세 이미지
│   ├── pose_data.csv          # 추출된 포즈 데이터
│   └── pose_statistics.csv    # 통계 데이터
├── models/                     # 저장된 모델들
│   ├── cnn_lstm_model.h5      # 훈련된 CNN+LSTM 모델
│   ├── scaler.pkl             # 데이터 스케일러
│   ├── label_encoder.pkl      # 라벨 인코더
│   └── model_metadata.json    # 모델 메타데이터
├── src/                       # 소스 코드
│   ├── preprocessing.py       # 1단계: 이미지 전처리
│   ├── cnn_lstm_model.py     # 2단계: CNN+LSTM 모델
│   ├── realtime_cam.py       # 3단계: 실시간 웹캠 시스템
│   └── analyze_upper_body.py # 상체 분석 유틸리티
├── app_ui/                    # PyQt6 GUI 애플리케이션
│   ├── main.py               # GUI 메인 실행
│   ├── main_window.py        # 메인 윈도우
│   ├── monitor_thread.py     # 모니터링 스레드
│   └── tabs/                 # 탭별 UI 모듈
├── run_system.py             # 통합 실행 스크립트
├── requirements.txt          # 의존성 패키지
└── README.md                 # 이 파일
```

## 🚀 빠른 시작

### 1. 환경 설정

```powershell
cd "C:\Users\user\OneDrive\Desktop\test\model_num1"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 통합 실행 (권장)

```powershell
# 샘플 데이터 폴더 생성
.\.venv\Scripts\python.exe run_system.py --setup

# 전체 시스템 실행 (전처리 → 훈련 → 실시간 모니터링)
.\.venv\Scripts\python.exe run_system.py --step all
```

### 3. 단계별 실행

```powershell
# 1단계: 이미지 전처리
.\.venv\Scripts\python.exe run_system.py --step preprocess

# 2단계: CNN+LSTM 모델 훈련
.\.venv\Scripts\python.exe run_system.py --step train

# 3단계: 실시간 웹캠 실행
.\.venv\Scripts\python.exe run_system.py --step monitor
```

## 📋 개발 단계

### 1단계: 이미지 전처리 (preprocessing.py)

**목적**: 이미지에서 자세 특성 추출 및 CSV 생성

**기능**:
- MediaPipe를 이용한 스켈레톤 추출
- 상체 랜드마크 (목, 어깨, 허리) 좌표 추출
- 자세 각도 계산 (목 기울기, 어깨 기울기, 허리 기울기, 상체 기울기)
- 통계 계산 (평균, 분산, 표준편차)
- 라벨별 분석 (normal/abnormal)

**출력**:
- `pose_data.csv`: 모든 이미지의 포즈 데이터
- `pose_statistics.csv`: 통계 정보

### 2단계: CNN+LSTM 하이브리드 모델 (cnn_lstm_model.py)

**목적**: 이미지와 시계열 데이터를 결합한 딥러닝 모델

**특징**:
- **TimeDistributed CNN**: 각 프레임에서 공간적 특징 추출
- **이중 LSTM 브랜치**: 
  - 이미지 시퀀스 처리 (LSTM 64→32)
  - 숫자 특성 시퀀스 처리 (LSTM 32→16)
- **클래스 가중치 균형**: 데이터 불균형 처리
- **Dropout과 BatchNormalization**: 과적합 방지
- **EarlyStopping과 학습률 조절**: 최적 학습

**모델 구조**:
```
[이미지 입력] → TimeDistributed(Conv2D×3) → LSTM×2 → Dense
[숫자 입력] → LSTM×2 → Dense
[융합] → Dense×2 → Softmax 출력
```

**파라미터**:
- 이미지 크기: 128×128
- 시퀀스 길이: 3 프레임
- 총 파라미터: ~169K
- Dropout: 0.3

### 3단계: 실시간 웹캠 시스템 (realtime_cam.py)

**목적**: 웹캠을 통한 실시간 자세 모니터링

**기능**:
- 실시간 웹캠 입력 처리
- MediaPipe로 실시간 포즈 추출
- 학습된 CNN+LSTM 모델로 자세 예측
- 시각적 피드백 (포즈 랜드마크, 상태 표시)
- 자세 교정 알림 시스템
- 예측 결과 스무딩 (노이즈 제거)

**제어**:
- `ESC`: 종료
- `R`: 시스템 리셋

## 🎨 GUI 애플리케이션

```powershell
# PyQt6 기반 GUI 실행
.\.venv\Scripts\python.exe app_ui/main.py
```

**기능**:
- 실시간 웹캠 모니터링
- 자세 통계 및 랭킹
- 프로필 관리
- 튜토리얼

## 🎯 사용법

### 데이터 준비

1. `data/train_images/normal/`에 정상 자세 이미지 추가
2. `data/train_images/abnormal/`에 비정상 자세 이미지 추가
3. 최소 각 클래스당 20장 이상 권장

### 모델 훈련

```powershell
# 직접 모델 훈련 (고급 옵션)
.\.venv\Scripts\python.exe src/cnn_lstm_model.py --epochs 80 --batch_size 4 --sequence_length 3
```

### 실시간 모니터링

```powershell
# 웹캠 모니터링 실행
.\.venv\Scripts\python.exe src/realtime_cam.py --camera 0
```

## 📊 성능 최적화

### 모델 파라미터 조정

- `sequence_length`: 시퀀스 길이 (기본: 3)
- `img_height`, `img_width`: 이미지 크기 (기본: 128×128)
- `epochs`: 훈련 에폭 (기본: 80)
- `batch_size`: 배치 크기 (기본: 4)
- `dropout_rate`: 드롭아웃 비율 (기본: 0.3)

### 데이터 품질 향상

- 다양한 각도의 이미지 수집
- 조명 조건 다양화
- 충분한 데이터량 확보 (클래스당 100장 이상)

## 🔧 문제 해결

### 일반적인 문제

- **카메라 접근 오류**: 다른 프로그램에서 카메라 사용 중인지 확인
- **모델 로드 실패**: 모델 파일 경로와 의존성 파일 확인
- **낮은 정확도**: 훈련 데이터 품질과 양 확인

### 성능 개선

- GPU 사용 설정 (CUDA 설치)
- 더 많은 훈련 데이터 수집
- 하이퍼파라미터 튜닝

## 📈 확장 가능성

### 추가 기능

- 더 많은 자세 클래스 (slouching, leaning, etc.)
- 음성 알림 시스템
- 웹 인터페이스 개발
- 자세 기록 및 분석 리포트
- 모바일 앱 연동

### 고급 기능

- 3D 포즈 추정
- 실시간 자세 교정 가이드
- 개인별 맞춤 임계값 설정
- 장시간 자세 패턴 분석

## 💡 현재 환경 상태

- ✅ Python 3.10.1 가상환경 활성화됨
- ✅ 모든 의존성 패키지 설치 완료
- ✅ CNN+LSTM 모델 훈련 완료
- ✅ 바로 실행 가능한 상태

## 📞 지원

문제가 발생하면 각 단계별 로그를 확인하고, 데이터와 모델 파일 상태를 점검해주세요.

---

**개발 환경**: Python 3.10.1 | TensorFlow/Keras | MediaPipe | OpenCV | PyQt6

