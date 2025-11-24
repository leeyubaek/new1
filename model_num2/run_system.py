"""
통합 실행 및 테스트 스크립트
"""

import os
import sys
import logging
from pathlib import Path
import argparse

# 현재 디렉토리를 파이썬 패스에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 로깅 설정
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def setup_sample_data():
    """샘플 데이터 설정"""
    data_dir = Path('data/train_images')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 샘플 폴더 구조 생성
    normal_dir = data_dir / 'normal'
    abnormal_dir = data_dir / 'abnormal'
    
    normal_dir.mkdir(exist_ok=True)
    abnormal_dir.mkdir(exist_ok=True)
    
    logging.info("샘플 데이터 폴더 구조 생성 완료")
    logging.info(f"정상 자세 이미지를 {normal_dir}에 넣어주세요")
    logging.info(f"비정상 자세 이미지를 {abnormal_dir}에 넣어주세요")
    
    return data_dir

def run_preprocessing(data_dir, labeled=True):
    """1단계: 전처리 실행"""
    logging.info("=== 1단계: 이미지 전처리 시작 ===")
    
    from src.preprocessing import PosePreprocessor
    
    preprocessor = PosePreprocessor()
    
    try:
        if labeled:
            # 라벨링된 데이터 처리
            all_data = preprocessor.process_labeled_data(str(data_dir))
        else:
            # 단일 폴더 처리
            all_data = preprocessor.process_image_folder(str(data_dir))
        
        if all_data:
            # 데이터 저장
            preprocessor.save_data(all_data, 'data')
            logging.info(f"전처리 완료: 총 {len(all_data)}개 이미지 처리")
            return True
        else:
            logging.warning("처리된 데이터가 없습니다.")
            return False
            
    except Exception as e:
        logging.error(f"전처리 중 오류 발생: {e}")
        return False

def check_and_update_preprocessing():
    """이미지 폴더와 CSV 동기화 확인 및 자동 갱신"""
    from pathlib import Path
    import os
    
    image_folder = Path('data/train_images')
    csv_file = Path('data/pose_data.csv')
    
    # 이미지 파일 개수 확인
    image_count = 0
    if image_folder.exists():
        for subfolder in ['normal', 'abnormal']:
            subfolder_path = image_folder / subfolder
            if subfolder_path.exists():
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
                for ext in image_extensions:
                    image_count += len(list(subfolder_path.glob(ext)))
                    image_count += len(list(subfolder_path.glob(ext.upper())))
    
    # CSV 파일 라인 수 확인
    csv_count = 0
    if csv_file.exists():
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            csv_count = sum(1 for line in f) - 1  # 헤더 제외
    
    logging.info(f"이미지 폴더: {image_count}개, CSV 데이터: {csv_count}개")
    
    # 불일치 또는 CSV 없으면 전처리 실행
    if not csv_file.exists() or image_count != csv_count:
        logging.warning(f"데이터 불일치 감지! 자동으로 전처리를 실행합니다...")
        if run_preprocessing(image_folder, labeled=True):
            logging.info("전처리 완료. 훈련을 계속합니다.")
            return True
        else:
            logging.error("전처리 실패")
            return False
    
    return True

def run_training():
    """2단계: CNN+LSTM 모델 훈련"""
    logging.info("=== 2단계: CNN+LSTM 모델 훈련 시작 ===")
    
    from src.cnn_lstm_model import CNNLSTMModel
    
    # 모델 저장 폴더 생성
    Path('models').mkdir(exist_ok=True)
    
    # 이미지와 CSV 동기화 확인 및 자동 갱신
    if not check_and_update_preprocessing():
        return False
    
    # CNN+LSTM 모델 초기화
    cnn_lstm_model = CNNLSTMModel(
        img_height=128, 
        img_width=128, 
        sequence_length=5,
        dropout_rate=0.4
    )
    
    try:
        # 데이터 로드
        logging.info("데이터 로드 중...")
        df, image_dir = cnn_lstm_model.load_data(
            'data/pose_data.csv',
            'data/train_images'
        )
        
        if len(df) == 0:
            logging.error("훈련 데이터가 없습니다.")
            return False
        
        # 데이터 전처리
        logging.info("데이터 전처리 중...")
        (X_img_train, X_num_train, y_train,
         X_img_val, X_num_val, y_val,
         X_img_test, X_num_test, y_test) = cnn_lstm_model.prepare_data(
            df, image_dir, augment=True, save_split=False
        )
        
        if len(X_img_train) == 0:
            logging.error("훈련 데이터가 없습니다.")
            return False
        
        # 모델 구성
        logging.info("모델 구성 중...")
        num_classes = len(cnn_lstm_model.label_encoder.classes_)
        num_numeric_features = len(cnn_lstm_model.feature_columns)
        
        cnn_lstm_model.build_model(num_numeric_features, num_classes)
        
        # 모델 훈련
        logging.info("모델 훈련 시작...")
        history = cnn_lstm_model.train(
            X_img_train, X_num_train, y_train,
            X_img_val, X_num_val, y_val,
            epochs=10,
            batch_size=4
        )
        
        # 훈련 히스토리 그래프 생성
        logging.info("훈련 히스토리 그래프 생성 중...")
        cnn_lstm_model.plot_training_history(history)
        
        # 모델 평가
        logging.info("훈련 세트 평가 중...")
        train_accuracy, _, _, _ = cnn_lstm_model.evaluate(
            X_img_train, X_num_train, y_train, dataset_name="Train"
        )
        
        logging.info("검증 세트 평가 중...")
        val_accuracy, _, _, _ = cnn_lstm_model.evaluate(
            X_img_val, X_num_val, y_val, dataset_name="Validation"
        )
        
        logging.info("테스트 세트 평가 중...")
        test_accuracy, _, _, _ = cnn_lstm_model.evaluate(
            X_img_test, X_num_test, y_test, dataset_name="Test"
        )
        
        # 모델 저장
        logging.info("모델 저장 중...")
        cnn_lstm_model.save_model()
        
        logging.info("CNN+LSTM 모델 훈련 완료!")
        logging.info(f"최종 훈련 정확도: {train_accuracy:.4f}")
        logging.info(f"최종 검증 정확도: {val_accuracy:.4f}")
        logging.info(f"최종 테스트 정확도: {test_accuracy:.4f}")
        return True
        
    except Exception as e:
        logging.error(f"훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_realtime_monitor():
    """3단계: 실시간 웹캠 모니터링"""
    logging.info("=== 3단계: 실시간 웹캠 모니터링 시작 ===")
    
    from src.realtime_cam import RealtimePostureMonitor
    
    try:
        # 모니터링 시스템 초기화
        monitor = RealtimePostureMonitor(model_path='models/cnn_lstm_model.h5')
        
        # 실행
        monitor.run(camera_id=0, show_window=True)
        
        return True
        
    except Exception as e:
        logging.error(f"실시간 모니터링 중 오류 발생: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='자세 교정 AI 시스템')
    parser.add_argument('--step', choices=['all', 'preprocess', 'train', 'monitor'], 
                       default='all', help='실행할 단계 선택')
    parser.add_argument('--setup', action='store_true', 
                       help='샘플 데이터 폴더 설정')
    
    args = parser.parse_args()
    
    # 현재 디렉토리 확인
    os.chdir(Path(__file__).parent)
    
    try:
        if args.setup:
            setup_sample_data()
            return
        
        success = True
        
        if args.step in ['all', 'preprocess']:
            # 데이터 폴더 확인
            data_dir = Path('data/train_images')
            if not data_dir.exists():
                data_dir = setup_sample_data()
            
            success = run_preprocessing(data_dir, labeled=True)
            if not success:
                logging.error("전처리 실패")
                return
        
        if args.step in ['all', 'train'] and success:
            # CSV 파일 확인
            if not Path('data/pose_data.csv').exists():
                logging.error("전처리 데이터가 없습니다. 먼저 전처리를 실행하세요.")
                return
            
            success = run_training()
            if not success:
                logging.error("모델 훈련 실패")
                return
        
        if args.step in ['all', 'monitor'] and success:
            # 모델 파일 확인
            if not Path('models/cnn_lstm_model.h5').exists():
                logging.error("훈련된 모델이 없습니다. 먼저 모델을 훈련하세요.")
                return
            
            run_realtime_monitor()
        
        if success:
            logging.info("🎉 모든 단계가 성공적으로 완료되었습니다!")
        
    except KeyboardInterrupt:
        logging.info("사용자에 의해 중단됨")
    except Exception as e:
        logging.error(f"실행 중 오류 발생: {e}")

if __name__ == '__main__':
    main()