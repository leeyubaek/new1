# 🚀 적용된 기술 및 알고리즘

## 📋 목차
1. [딥러닝 아키텍처](#딥러닝-아키텍처)
2. [컴퓨터 비전](#컴퓨터-비전)
3. [데이터 전처리](#데이터-전처리)
4. [최적화 기법](#최적화-기법)
5. [정규화 기법](#정규화-기법)
6. [평가 메트릭](#평가-메트릭)

---

## 🧠 딥러닝 아키텍처

### 1. **CNN (Convolutional Neural Network)**
- **역할**: 이미지에서 공간적 특징 추출
- **적용 위치**: TimeDistributed CNN 레이어
- **구성**:
  - Conv2D 32 → 64 → 128 필터
  - 3×3 커널 사이즈
  - ReLU 활성화 함수
  - MaxPooling2D (2×2)
  - GlobalAveragePooling2D

```python
Conv2D(32, (3,3)) → Conv2D(64, (3,3)) → Conv2D(128, (3,3))
```

### 2. **LSTM (Long Short-Term Memory)**
- **역할**: 시계열 패턴 학습 및 장기 의존성 포착
- **적용**:
  - 이미지 시퀀스: LSTM(64) → LSTM(32)
  - 수치 시퀀스: LSTM(32) → LSTM(16)
- **특징**:
  - Bidirectional 미사용 (단방향)
  - return_sequences=True (첫 번째 레이어)
  - Dropout 0.4 적용
  - Recurrent Dropout 0.2 적용

### 3. **TimeDistributed**
- **역할**: 시퀀스의 각 타임스텝에 동일한 레이어 적용
- **적용**: CNN 레이어를 각 프레임에 독립적으로 적용
- **장점**: 파라미터 공유로 효율성 증가

### 4. **Hybrid Architecture (멀티모달)**
- **구조**: 이중 브랜치 아키텍처
  ```
  [이미지 브랜치]        [수치 브랜치]
  TimeDistributed CNN    LSTM
         ↓                  ↓
      LSTM × 2          LSTM × 2
         ↓                  ↓
      Dense 32          Dense 16
         └────── Fusion ──────┘
                  ↓
            Dense 32 → 16
                  ↓
              Softmax
  ```

---

## 👁️ 컴퓨터 비전

### 1. **MediaPipe Pose Estimation**
- **기술**: Google MediaPipe의 포즈 추정 모델
- **적용**:
  - 33개 랜드마크 중 상체 5개 추출
  - nose, left_shoulder, right_shoulder, left_hip, right_hip
- **설정**:
  - model_complexity: 1 (균형)
  - min_detection_confidence: 0.6
  - static_image_mode: True

### 2. **Skeleton-based Pose Analysis**
- **기술**: 스켈레톤 기반 자세 분석
- **계산 특징**:
  - 목 각도 (Neck Angle)
  - 어깨 기울기 (Shoulder Angle)
  - 허리 기울기 (Hip Angle)
  - 상체 기울기 (Torso Angle)
  - 추가: shoulder_width, hip_width, torso_height, neck_length

### 3. **Angle Normalization**
- **기술**: 각도 정규화 (-180° ~ 180°)
- **목적**: 일관된 각도 표현

---

## 🔧 데이터 전처리

### 1. **Data Augmentation (데이터 증강)**
- **이미지 증강**:
  - Rotation: ±5°
  - Width/Height Shift: ±5%
  - Zoom: ±5%
- **수치 증강**:
  - Gaussian Noise: μ=0, σ=0.005
- **증강 팩터**: 1 (원본 × 2배)

### 2. **Sequence Generation**
- **기술**: 슬라이딩 윈도우 방식
- **설정**: sequence_length = 5 (기본값)
- **목적**: 시계열 패턴 학습

### 3. **Data Normalization**
- **StandardScaler**: 수치 데이터 정규화 (평균 0, 분산 1)
- **Image Normalization**: 픽셀 값을 0~1로 정규화 (÷255)

### 4. **Train-Validation-Test Split**
- **비율**: 70% / 20% / 10%
- **Stratified Split**: 클래스 비율 유지
- **random_state=42**: 재현 가능성 보장

---

## ⚙️ 최적화 기법

### 1. **Adam Optimizer**
- **Learning Rate**: 0.0005 (감소됨)
- **Gradient Clipping**: clipnorm=1.0
- **특징**: 
  - Adaptive learning rate
  - Momentum 포함

### 2. **Learning Rate Scheduling**
- **ReduceLROnPlateau**:
  - monitor: val_loss
  - factor: 0.6 (40% 감소)
  - patience: 10 epochs
  - min_lr: 1e-7
  - min_delta: 0.001

### 3. **Early Stopping**
- **monitor**: val_loss
- **patience**: 25 epochs
- **restore_best_weights**: True
- **min_delta**: 0.001

### 4. **Class Weight Balancing**
- **기술**: compute_class_weight('balanced')
- **목적**: 클래스 불균형 해결
- **적용**: 훈련 시 class_weight 파라미터

---

## 🛡️ 정규화 기법

### 1. **Dropout**
- **비율**: 0.4 (기본)
- **적용 위치**:
  - TimeDistributed CNN: 0.2 (50% 감소)
  - LSTM: 0.4
  - Dense layers: 0.4

### 2. **Recurrent Dropout**
- **비율**: 0.2
- **적용**: 모든 LSTM 레이어
- **목적**: LSTM 내부 과적합 방지

### 3. **L2 Regularization**
- **계수**: 0.001
- **적용 레이어**:
  - Conv2D 레이어
  - LSTM 레이어
  - Dense 레이어
- **목적**: 가중치 크기 제한

### 4. **Batch Normalization**
- **Momentum**: 0.9
- **적용 위치**:
  - 각 Conv2D 레이어 후
  - 각 LSTM 레이어 후
  - Dense 레이어 후
- **효과**: 내부 공변량 변화 감소

---

## 📊 평가 메트릭

### 1. **분류 메트릭**
- **Accuracy** (정확도): 전체 정답률
- **Precision** (정밀도): 예측한 것 중 맞은 비율
- **Recall** (재현율): 실제 정답 중 찾아낸 비율
- **F1-Score**: Precision과 Recall의 조화평균

### 2. **Confusion Matrix** (혼동행렬)
- **시각화**: Seaborn heatmap
- **표시 정보**: 개수 + 퍼센트
- **용도**: 클래스별 오분류 패턴 분석

### 3. **ROC & Confidence Analysis**
- **평균 예측 신뢰도**: 모델의 확신도 측정
- **클래스별 신뢰도**: 각 클래스의 예측 신뢰도

### 4. **Class Performance Visualization**
- Precision, Recall, F1-Score 비교 그래프
- 샘플 분포 및 신뢰도 그래프

---

## 🔬 추가 기술

### 1. **한글 경로 지원**
```python
np.fromfile() + cv2.imdecode()
```
- **목적**: Windows 한글 경로 문제 해결

### 2. **Visibility Filtering**
- **임계값**: 0.5
- **목적**: 품질 낮은 랜드마크 제거

### 3. **Image Quality Validation**
- **최소 크기**: 100×100 픽셀
- **목적**: 너무 작은 이미지 필터링

### 4. **Progress Tracking**
- 10% 단위 진행률 표시
- 성공/실패 카운트 추적

---

## 📦 사용된 주요 라이브러리

### Deep Learning
- **TensorFlow/Keras**: 2.18.0
  - 모델 구축 및 훈련
  - 레이어, 옵티마이저, 콜백

### Computer Vision
- **MediaPipe**: 0.10.21
  - 포즈 추정
- **OpenCV**: 4.11.0
  - 이미지 처리

### Data Processing
- **NumPy**: 배열 연산
- **Pandas**: 2.3.3 - 데이터 처리
- **scikit-learn**: 전처리, 평가

### Visualization
- **Matplotlib**: 그래프 생성
- **Seaborn**: 고급 시각화

### GUI
- **PyQt6**: 6.10.0 - GUI 애플리케이션

---

## 🎯 핵심 혁신 포인트

1. **Hybrid Multi-Modal Architecture**
   - 이미지 + 수치 데이터 융합
   - 공간적 특징 + 시계열 패턴

2. **Temporal Sequence Learning**
   - LSTM을 통한 동작 패턴 학습
   - 단일 프레임이 아닌 연속 프레임 분석

3. **Robust Regularization**
   - Dropout + L2 + Batch Normalization
   - Recurrent Dropout으로 LSTM 안정화

4. **Smart Data Augmentation**
   - 훈련 데이터만 증강
   - 미세한 변형으로 안정성 유지

5. **Comprehensive Evaluation**
   - 다양한 메트릭
   - 시각적 성능 분석
   - 과적합 자동 감지

---

## 📈 성능 최적화 전략

### 메모리 최적화
- GlobalAveragePooling2D 사용
- 적절한 배치 크기 (8-16)
- 이미지 크기 128×128

### 학습 안정화
- Gradient Clipping
- Learning Rate Scheduling
- Early Stopping
- BatchNormalization

### 과적합 방지
- Multiple Regularization
- Data Augmentation (제한적)
- Class Balancing
- Validation Monitoring

---

## 🔍 모델 아키텍처 요약

```
입력:
├─ 이미지 시퀀스: (5, 128, 128, 3)
└─ 수치 시퀀스: (5, 4~14) features

처리:
├─ CNN Branch:
│  └─ TimeDistributed(Conv2D×3 + BN + Dropout)
│     → LSTM(64) → LSTM(32) → Dense(32)
│
└─ Numeric Branch:
   └─ LSTM(32) → LSTM(16) → Dense(16)

융합:
└─ Concatenate → Dense(32) → Dense(16)

출력:
└─ Softmax (클래스 확률)
```

**총 파라미터**: ~168,000개
- Trainable: ~167,200
- Non-trainable: ~800 (BatchNorm)

---

이 프로젝트는 **최신 딥러닝 기법**과 **컴퓨터 비전 기술**을 결합하여 실시간 자세 분석이 가능한 강건한 시스템을 구현했습니다! 🎉
