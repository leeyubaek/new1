@echo off
chcp 65001 > nul
echo 🤖 웹캠 기반 실시간 자세 판단 AI 시스템
echo ================================================
echo.

cd /d "%~dp0"

:menu
cls
echo.
echo ================================================
echo    🤖 웹캠 기반 실시간 자세 판단 AI 시스템
echo ================================================
echo.
echo 🎯 실행 옵션을 선택하세요:
echo.
echo [1] 📂 데이터 폴더 생성
echo [2] 🚀 전체 시스템 실행 (전처리 → 훈련 → 모니터링)
echo [3] 📊 1단계: 이미지 전처리만
echo [4] 🧠 2단계: 모델 훈련만  
echo [5] 📹 3단계: 실시간 모니터링만
echo [6] 🔧 개별 스크립트 메뉴
echo [7] ❓ 도움말
echo [0] 🚪 종료
echo.
echo ================================================

set /p choice="번호를 입력하세요 (0-7): "

if "%choice%"=="1" goto setup
if "%choice%"=="2" goto run_all
if "%choice%"=="3" goto preprocess
if "%choice%"=="4" goto train
if "%choice%"=="5" goto monitor
if "%choice%"=="6" goto individual
if "%choice%"=="7" goto help
if "%choice%"=="0" goto exit
goto invalid

:setup
cls
echo.
echo ================================================
echo    📂 데이터 폴더 생성
echo ================================================
echo.
.\.venv\Scripts\python.exe run_system.py --setup
echo.
echo ✅ 완료! 
echo.
echo 다음 폴더에 이미지를 넣어주세요:
echo   - data/train_images/normal/    (정상 자세 이미지)
echo   - data/train_images/abnormal/  (비정상 자세 이미지)
echo.
pause
goto menu

:run_all
cls
echo.
echo ================================================
echo    🚀 전체 시스템 실행
echo ================================================
echo.
echo 실행 순서:
echo   1단계: 이미지 전처리
echo   2단계: 모델 훈련 (30 에폭)
echo   3단계: 실시간 모니터링
echo.
echo 시작합니다...
echo.
.\.venv\Scripts\python.exe run_system.py --step all
echo.
echo ================================================
pause
goto menu

:preprocess
cls
echo.
echo ================================================
echo    📊 1단계: 이미지 전처리
echo ================================================
echo.
echo MediaPipe를 이용하여 이미지에서 포즈 데이터를 추출합니다.
echo 결과는 data/pose_data.csv에 저장됩니다.
echo.
.\.venv\Scripts\python.exe run_system.py --step preprocess
echo.
echo ================================================
echo ✅ 전처리 완료!
echo    - 결과: data/pose_data.csv
echo    - 통계: data/pose_statistics.csv
echo ================================================
echo.
pause
goto menu

:train
cls
echo.
echo ================================================
echo    🧠 2단계: CNN+LSTM 모델 훈련
echo ================================================
echo.
echo 설정:
echo   - 모델: CNN + LSTM
echo   - 에폭: 30
echo   - 배치 크기: 4
echo   - 데이터 증강: 3배
echo.
echo 참고: 데이터 양에 따라 시간이 걸릴 수 있습니다.
echo.
.\.venv\Scripts\python.exe run_system.py --step train
echo.
echo ================================================
echo ✅ 훈련 완료!
echo    - 모델: models/cnn_lstm_model.h5
echo    - 그래프: models/ 폴더 확인
echo ================================================
echo.
pause
goto menu

:monitor
cls
echo.
echo ================================================
echo    📹 3단계: 실시간 웹캠 모니터링
echo ================================================
echo.
echo 조작법:
echo   - ESC: 종료
echo   - R: 카운터 리셋
echo.
echo 웹캠을 시작합니다...
echo.
.\.venv\Scripts\python.exe run_system.py --step monitor
echo.
pause
goto menu

:individual
cls
echo.
echo ================================================
echo    🔧 개별 스크립트 실행 메뉴
echo ================================================
echo.
echo [1] 📊 전처리 스크립트 (고급 옵션)
echo [2] 🧠 CNN+LSTM 모델 스크립트 (고급 옵션)
echo [3] 📹 실시간 웹캠 스크립트 (고급 옵션)
echo [4] ⚡ 빠른 테스트 훈련 (10 에폭)
echo [5] 🔬 긴 훈련 (50 에폭)
echo [0] 🔙 메인 메뉴로 돌아가기
echo.
echo ================================================

set /p sub_choice="번호를 입력하세요 (0-5): "

if "%sub_choice%"=="1" goto advanced_preprocess
if "%sub_choice%"=="2" goto advanced_train
if "%sub_choice%"=="3" goto advanced_monitor
if "%sub_choice%"=="4" goto quick_train
if "%sub_choice%"=="5" goto long_train
if "%sub_choice%"=="0" goto menu
goto individual

:advanced_preprocess
cls
echo.
echo ================================================
echo    📊 고급 전처리 실행
echo ================================================
echo.
.\.venv\Scripts\python.exe src/preprocessing.py --input data/train_images --labeled --output data
echo.
echo ================================================
pause
goto individual

:advanced_train
cls
echo.
echo ================================================
echo    🧠 고급 모델 훈련 (30 에폭)
echo ================================================
echo.
.\.venv\Scripts\python.exe src/cnn_lstm_model.py --data data/pose_data.csv --images data/train_images --epochs 30 --batch_size 4
echo.
echo ================================================
pause
goto individual

:advanced_monitor
cls
echo.
echo ================================================
echo    📹 고급 실시간 모니터링
echo ================================================
echo.
.\.venv\Scripts\python.exe src/realtime_cam.py --camera 0
echo.
echo ================================================
pause
goto individual

:quick_train
cls
echo.
echo ================================================
echo    ⚡ 빠른 테스트 훈련 (10 에폭)
echo ================================================
echo.
echo 빠른 테스트를 위한 짧은 훈련입니다.
echo.
.\.venv\Scripts\python.exe src/cnn_lstm_model.py --data data/pose_data.csv --images data/train_images --epochs 10 --batch_size 4
echo.
echo ================================================
pause
goto individual

:long_train
cls
echo.
echo ================================================
echo    🔬 긴 훈련 (50 에폭)
echo ================================================
echo.
echo 더 정확한 모델을 위한 긴 훈련입니다.
echo 완료까지 시간이 오래 걸릴 수 있습니다.
echo.
.\.venv\Scripts\python.exe src/cnn_lstm_model.py --data data/pose_data.csv --images data/train_images --epochs 50 --batch_size 4
echo.
echo ================================================
pause
goto individual

:help
cls
echo.
echo ================================================
echo    ❓ 도움말
echo ================================================
echo.
echo 🎯 사용 순서:
echo ────────────────────────────────────────────────
echo 1. [1] 데이터 폴더 생성
echo    └─→ data/train_images/ 폴더 생성
echo.
echo 2. 이미지 준비
echo    ├─→ data/train_images/normal/    (정상 자세)
echo    └─→ data/train_images/abnormal/  (비정상 자세)
echo.
echo 3. [2] 전체 시스템 실행
echo    ├─→ 자동으로 전처리 실행
echo    ├─→ 모델 훈련
echo    └─→ 실시간 모니터링
echo.
echo 📂 필요한 이미지:
echo ────────────────────────────────────────────────
echo - 각 폴더당 최소 10장 이상
echo - 다양한 각도와 조명 조건
echo - 선명하고 전신이 보이는 이미지
echo - JPG, JPEG, PNG, BMP 형식 지원
echo.
echo 📊 생성되는 파일:
echo ────────────────────────────────────────────────
echo - data/pose_data.csv           : 포즈 데이터
echo - data/pose_statistics.csv     : 통계 정보
echo - models/cnn_lstm_model.h5     : 훈련된 모델
echo - models/training_history.png  : 훈련 그래프
echo - models/confusion_matrix_*.png: 혼동행렬
echo - models/class_performance_*.png: 성능 분석
echo.
echo 🔧 문제 해결:
echo ────────────────────────────────────────────────
echo - 카메라 오류
echo   └─→ 다른 프로그램에서 카메라 사용 중인지 확인
echo.
echo - 모델 없음 오류
echo   └─→ 먼저 [4] 모델 훈련을 실행하세요
echo.
echo - 낮은 정확도
echo   └─→ 더 많은 데이터를 추가하세요
echo   └─→ [6-5] 긴 훈련(50 에폭)을 시도하세요
echo.
echo - CSV 데이터 불일치
echo   └─→ 자동으로 재전처리됩니다
echo.
echo 📖 자세한 내용:
echo ────────────────────────────────────────────────
echo - README.md 파일 참고
echo - EXECUTION_GUIDE.md 파일 참고
echo.
echo ================================================
pause
goto menu

:invalid
cls
echo.
echo ================================================
echo    ❌ 잘못된 입력
echo ================================================
echo.
echo 0-7 사이의 숫자를 입력하세요.
echo.
timeout /t 2 > nul
goto menu

:exit
cls
echo.
echo ================================================
echo    👋 프로그램을 종료합니다
echo ================================================
echo.
echo 감사합니다!
echo.
timeout /t 2 > nul
exit
