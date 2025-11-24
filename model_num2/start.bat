@echo off
chcp 65001 > nul
echo 🤖 웹캠 기반 실시간 자세 판단 AI 시스템
echo ================================================
echo.

cd /d "%~dp0"

:menu
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
echo.
echo 📂 데이터 폴더 생성 중...
echo.
.\.venv\Scripts\python.exe run_system.py --setup
echo.
echo ✅ 완료! data/train_images/normal/ 과 abnormal/ 폴더에 이미지를 넣어주세요.
pause
goto menu

:run_all
echo.
echo 🚀 전체 시스템 실행 중...
echo 단계: 전처리 → 모델 훈련 → 실시간 모니터링
echo.
.\.venv\Scripts\python.exe run_system.py --step all
pause
goto menu

:preprocess
echo.
echo 📊 1단계: 이미지 전처리 실행 중...
echo.
.\.venv\Scripts\python.exe run_system.py --step preprocess
echo.
echo ✅ 전처리 완료! data/pose_data.csv 파일이 생성되었습니다.
pause
goto menu

:train
echo.
echo 🧠 2단계: LSTM 모델 훈련 중...
echo 참고: 데이터 양에 따라 시간이 걸릴 수 있습니다.
echo.
.\.venv\Scripts\python.exe run_system.py --step train
echo.
echo ✅ 훈련 완료! models/ 폴더에 모델이 저장되었습니다.
pause
goto menu

:monitor
echo.
echo 📹 3단계: 실시간 웹캠 모니터링 시작
echo 조작법: ESC=종료, R=리셋
echo.
.\.venv\Scripts\python.exe run_system.py --step monitor
pause
goto menu

:individual
echo.
echo 🔧 개별 스크립트 실행 메뉴
echo ================================
echo.
echo [1] 전처리 스크립트 (고급 옵션)
echo [2] LSTM 모델 스크립트 (고급 옵션)
echo [3] 실시간 웹캠 스크립트 (고급 옵션)
echo [4] 빠른 테스트 훈련 (10 에폭)
echo [0] 메인 메뉴로 돌아가기
echo.

set /p sub_choice="번호를 입력하세요 (0-4): "

if "%sub_choice%"=="1" goto advanced_preprocess
if "%sub_choice%"=="2" goto advanced_train
if "%sub_choice%"=="3" goto advanced_monitor
if "%sub_choice%"=="4" goto quick_train
if "%sub_choice%"=="0" goto menu
goto individual

:advanced_preprocess
echo.
echo 📊 고급 전처리 실행 중...
.\.venv\Scripts\python.exe src/preprocessing.py --input data/train_images --labeled --output data
pause
goto individual

:advanced_train
echo.
echo 🧠 고급 모델 훈련 실행 중...
.\.venv\Scripts\python.exe src/cnn_lstm_model.py --data data/pose_data.csv --images data/train_images --epochs 100 --batch_size 32
pause
goto individual

:advanced_monitor
echo.
echo 📹 고급 실시간 모니터링 실행 중...
.\.venv\Scripts\python.exe src/realtime_cam.py --camera 0
pause
goto individual

:quick_train
echo.
echo ⚡ 빠른 테스트 훈련 (10 에폭)
.\.venv\Scripts\python.exe src/cnn_lstm_model.py --data data/pose_data.csv --images data/train_images --epochs 10 --batch_size 16
pause
goto individual

:help
echo.
echo ❓ 도움말
echo =========
echo.
echo 🎯 사용 순서:
echo 1. 먼저 "데이터 폴더 생성"으로 폴더를 만듭니다
echo 2. data/train_images/normal/ 에 정상 자세 이미지를 넣습니다
echo 3. data/train_images/abnormal/ 에 비정상 자세 이미지를 넣습니다
echo 4. "전체 시스템 실행"을 선택합니다
echo.
echo 📂 필요한 이미지:
echo - 각 폴더당 최소 10장 이상
echo - 다양한 각도와 조명 조건
echo - 선명하고 전신이 보이는 이미지
echo.
echo 🔧 문제 해결:
echo - 카메라 오류: 다른 프로그램에서 카메라 사용 중인지 확인
echo - 모델 없음: 먼저 훈련을 실행하세요
echo - 낮은 정확도: 더 많은 데이터를 추가하세요
echo.
echo 📖 자세한 내용은 EXECUTION_GUIDE.md 파일을 참고하세요.
echo.
pause
goto menu

:invalid
echo.
echo ❌ 잘못된 선택입니다. 0-7 사이의 숫자를 입력하세요.
echo.
pause
goto menu

:exit
echo.
echo 👋 프로그램을 종료합니다.
echo.
pause
exit