# 자세 분류 모델 - 최적화 버전

## 🚀 속도 최적화 기능

### 1. **병렬 이미지 로딩**
- ThreadPoolExecutor를 사용한 멀티스레드 이미지 로드
- 최대 8개의 워커로 병렬 처리
- **약 2-3배 빠른 데이터 로딩**

### 2. **TensorFlow Dataset 캐싱**
- 메모리 캐싱으로 반복적인 데이터 읽기 제거
- Auto-tuning prefetch로 자동 최적화
- **에폭당 학습 시간 단축**

### 3. **배치 크기 자동 최적화**
- GPU 감지 후 자동으로 배치 크기 조정
- GPU 사용 시: 배치 크기 x2 (최대 32)
- CPU 사용 시: 적정 배치 크기 유지
- **메모리 효율성 향상**

### 4. **Mixed Precision 학습 (GPU only)**
- FP16 연산으로 GPU 학습 속도 향상
- Loss Scaling으로 수치 안정성 보장
- CPU에서는 자동으로 비활성화
- **GPU 학습 속도 약 1.5-2배 향상**

### 5. **Early Stopping 최적화**
- Patience: 18 → 12로 단축
- 더 빠른 학습 종료 결정
- **불필요한 에폭 제거**

### 6. **학습률 스케줄링 개선**
- ReduceLROnPlateau patience: 7 → 5
- factor: 0.6 → 0.5로 더 공격적 감소
- **더 빠른 수렴**

## 📊 실행 방법

### 기본 학습 (최적화 적용)
```bash
.venv\Scripts\python.exe src\cnn_lstm_model.py --epochs 35 --batch_size 16
```

### 빠른 테스트 (5 에폭)
```bash
.venv\Scripts\python.exe src\cnn_lstm_model.py --epochs 5 --batch_size 16
```

### 옵션
- `--epochs`: 에폭 수 (기본: 35)
- `--batch_size`: 배치 크기 (기본: 16, 자동 최적화됨)
- `--img_size`: 이미지 크기 (기본: 112)
- `--no_augment`: 데이터 증강 비활성화

## ⚡ 예상 속도 향상

| 환경 | 기존 속도 | 최적화 후 | 개선율 |
|------|----------|----------|--------|
| CPU | 3-7 min/epoch | 2-4 min/epoch | **~40% 향상** |
| GPU | 30-60 sec/epoch | 15-30 sec/epoch | **~50% 향상** |

## 🔧 핵심 파일

- `src/cnn_lstm_model.py`: 메인 학습 스크립트 (최적화 적용)
- `src/model_utils.py`: 병렬 이미지 로딩
- `src/model_builder.py`: Mixed Precision 지원
- `src/preprocessing.py`: 데이터 전처리
- `src/realtime_cam.py`: 웹캠 실시간 적용

## 📝 주요 변경사항

### cnn_lstm_model.py
- TensorFlow Dataset API 사용 (캐싱 + prefetch)
- 자동 배치 크기 최적화
- GPU 감지 후 Mixed Precision 조건부 활성화
- Early Stopping patience 단축

### model_utils.py
- ThreadPoolExecutor로 병렬 이미지 로딩
- 최대 8개 워커로 동시 처리
- 진행률 실시간 표시

### model_builder.py
- Mixed Precision Loss Scaling 지원
- GPU 전용 최적화

## 💡 팁

1. **GPU 사용 시**: Mixed Precision이 자동 활성화되어 최대 성능 발휘
2. **CPU 사용 시**: 병렬 로딩과 캐싱으로 속도 개선
3. **배치 크기**: 자동 최적화되므로 기본값 사용 권장
4. **메모리 부족 시**: `--batch_size` 를 8로 낮추기

## 🎯 성능 모니터링

학습 시작 시 다음 정보가 표시됩니다:
```
✅ Mixed Precision (FP16) 활성화  # GPU 전용
최적화된 배치 크기: 32            # 자동 조정
진행률: 100% (544/636)           # 병렬 로딩
```

## ⚠️ 주의사항

- Mixed Precision은 GPU에서만 활성화됩니다
- CPU에서는 자동으로 비활성화되어 안정성 보장
- 배치 크기는 메모리와 데이터셋 크기에 따라 자동 조정됩니다
