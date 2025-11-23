"""
1단계: 이미지 전처리 코드
- MediaPipe를 이용한 스켈레톤 추출
- 상체 데이터(목, 어깨, 허리) CSV 저장
- 통계(평균, 분산) 계산
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import argparse

# 로깅 설정
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 상체 랜드마크 인덱스 정의
UPPER_BODY_LANDMARKS = {
    'nose': 0,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_hip': 23,
    'right_hip': 24
}

class PosePreprocessor:
    def __init__(self, model_complexity=1):
        """초기화 - MediaPipe Pose 설정 최적화"""
        self.pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,  # 0(가벼움), 1(균형), 2(정확)
            min_detection_confidence=0.6,  # 0.5 -> 0.6 (더 신뢰성 있는 감지)
            min_tracking_confidence=0.5
        )
        self.processed_count = 0
        self.failed_count = 0
        
    def safe_imread(self, path: Path) -> Optional[np.ndarray]:
        """안전한 이미지 읽기 (한글 경로 지원) + 품질 검증"""
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size == 0:
                return None
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            
            # 이미지 품질 검증
            if img is None:
                return None
            
            h, w = img.shape[:2]
            # 너무 작은 이미지 필터링 (최소 100x100)
            if h < 100 or w < 100:
                logging.warning(f"이미지 크기가 너무 작음 ({w}x{h}): {path.name}")
                return None
            
            return img
        except Exception as e:
            logging.error(f"이미지 로드 실패 {path}: {e}")
            return None
    
    def extract_pose_landmarks(self, image: np.ndarray) -> Optional[Dict]:
        """이미지에서 포즈 랜드마크 추출 + visibility 검증"""
        try:
            # BGR to RGB 변환
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_image)
            
            if not results.pose_landmarks:
                return None
                
            landmarks = results.pose_landmarks.landmark
            h, w = image.shape[:2]
            
            # 상체 랜드마크 추출 및 visibility 검증
            pose_data = {}
            min_visibility = 0.5  # 최소 가시성 임계값
            
            for name, idx in UPPER_BODY_LANDMARKS.items():
                landmark = landmarks[idx]
                
                # visibility가 너무 낮으면 제외
                if landmark.visibility < min_visibility:
                    logging.debug(f"{name} visibility 낮음: {landmark.visibility:.2f}")
                    return None
                
                pose_data[f'{name}_x'] = landmark.x
                pose_data[f'{name}_y'] = landmark.y
                pose_data[f'{name}_x_px'] = int(landmark.x * w)
                pose_data[f'{name}_y_px'] = int(landmark.y * h)
                pose_data[f'{name}_visibility'] = landmark.visibility
            
            # 각도 계산
            angles = self.calculate_angles(pose_data)
            pose_data.update(angles)
            
            # 추가 특성 계산
            extra_features = self.calculate_extra_features(pose_data)
            pose_data.update(extra_features)
            
            return pose_data
            
        except Exception as e:
            logging.error(f"포즈 추출 실패: {e}")
            return None
    
    def calculate_angles(self, pose_data: Dict) -> Dict:
        """자세 각도 계산 (개선된 버전)"""
        angles = {}
        
        try:
            # 목 각도 (코와 어깨 중심점 기준)
            nose_x, nose_y = pose_data['nose_x'], pose_data['nose_y']
            left_shoulder_x = pose_data['left_shoulder_x']
            left_shoulder_y = pose_data['left_shoulder_y']
            right_shoulder_x = pose_data['right_shoulder_x']
            right_shoulder_y = pose_data['right_shoulder_y']
            
            # 어깨 중심점
            shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2
            shoulder_center_y = (left_shoulder_y + right_shoulder_y) / 2
            
            # 목 기울기 각도
            neck_angle = np.degrees(np.arctan2(
                nose_x - shoulder_center_x,
                shoulder_center_y - nose_y
            ))
            angles['neck_angle'] = self._normalize_angle(neck_angle)
            
            # 어깨 기울기 각도
            shoulder_angle = np.degrees(np.arctan2(
                right_shoulder_y - left_shoulder_y,
                right_shoulder_x - left_shoulder_x
            ))
            angles['shoulder_angle'] = self._normalize_angle(shoulder_angle)
            
            # 허리 기울기 각도
            left_hip_x = pose_data['left_hip_x']
            left_hip_y = pose_data['left_hip_y']
            right_hip_x = pose_data['right_hip_x']
            right_hip_y = pose_data['right_hip_y']
            
            hip_angle = np.degrees(np.arctan2(
                right_hip_y - left_hip_y,
                right_hip_x - left_hip_x
            ))
            angles['hip_angle'] = self._normalize_angle(hip_angle)
            
            # 상체 전체 기울기 (어깨 중심 - 허리 중심)
            hip_center_x = (left_hip_x + right_hip_x) / 2
            hip_center_y = (left_hip_y + right_hip_y) / 2
            
            torso_angle = np.degrees(np.arctan2(
                shoulder_center_x - hip_center_x,
                hip_center_y - shoulder_center_y
            ))
            angles['torso_angle'] = self._normalize_angle(torso_angle)
            
        except Exception as e:
            logging.error(f"각도 계산 실패: {e}")
            # 기본값으로 0 설정
            angles = {
                'neck_angle': 0.0,
                'shoulder_angle': 0.0,
                'hip_angle': 0.0,
                'torso_angle': 0.0
            }
        
        return angles
    
    def _normalize_angle(self, angle: float) -> float:
        """각도를 -180 ~ 180 범위로 정규화"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    def calculate_extra_features(self, pose_data: Dict) -> Dict:
        """추가 특성 계산 (거리, 비율 등)"""
        features = {}
        
        try:
            # 어깨 너비
            shoulder_width = abs(pose_data['right_shoulder_x'] - pose_data['left_shoulder_x'])
            features['shoulder_width'] = shoulder_width
            
            # 엉덩이 너비
            hip_width = abs(pose_data['right_hip_x'] - pose_data['left_hip_x'])
            features['hip_width'] = hip_width
            
            # 상체 높이 (어깨 중심 - 엉덩이 중심)
            shoulder_center_y = (pose_data['left_shoulder_y'] + pose_data['right_shoulder_y']) / 2
            hip_center_y = (pose_data['left_hip_y'] + pose_data['right_hip_y']) / 2
            torso_height = abs(hip_center_y - shoulder_center_y)
            features['torso_height'] = torso_height
            
            # 목 길이 (코 - 어깨 중심)
            neck_length = np.sqrt(
                (pose_data['nose_x'] - (pose_data['left_shoulder_x'] + pose_data['right_shoulder_x'])/2)**2 +
                (pose_data['nose_y'] - (pose_data['left_shoulder_y'] + pose_data['right_shoulder_y'])/2)**2
            )
            features['neck_length'] = neck_length
            
            # 비율 계산
            if hip_width > 0:
                features['shoulder_hip_ratio'] = shoulder_width / hip_width
            else:
                features['shoulder_hip_ratio'] = 1.0
                
        except Exception as e:
            logging.debug(f"추가 특성 계산 실패: {e}")
            features = {
                'shoulder_width': 0.0,
                'hip_width': 0.0,
                'torso_height': 0.0,
                'neck_length': 0.0,
                'shoulder_hip_ratio': 1.0
            }
        
        return features
    
    def process_image_folder(self, input_folder: str, label: str = 'unlabeled') -> List[Dict]:
        """폴더 내 모든 이미지 처리"""
        input_path = Path(input_folder)
        if not input_path.exists():
            logging.error(f"폴더가 존재하지 않습니다: {input_folder}")
            return []
        
        # 이미지 파일 찾기
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(ext))
            image_files.extend(input_path.glob(ext.upper()))
        
        if not image_files:
            logging.warning(f"이미지 파일을 찾을 수 없습니다: {input_folder}")
            return []
        
        results = []
        processed_count = 0
        failed_count = 0
        total_files = len(image_files)
        
        logging.info(f"총 {total_files}개 이미지 처리 시작...")
        
        for idx, image_file in enumerate(image_files, 1):
            image = self.safe_imread(image_file)
            if image is None:
                failed_count += 1
                continue
            
            pose_data = self.extract_pose_landmarks(image)
            if pose_data is None:
                logging.warning(f"포즈 추출 실패: {image_file.name}")
                failed_count += 1
                continue
            
            # 이미지 정보 추가
            pose_data['image_name'] = image_file.name
            pose_data['image_path'] = str(image_file)
            pose_data['label'] = label
            
            results.append(pose_data)
            processed_count += 1
            
            # 진행률 표시 개선 (10% 단위)
            if idx % max(1, total_files // 10) == 0 or idx == total_files:
                progress = (idx / total_files) * 100
                logging.info(f"진행률: {progress:.1f}% ({idx}/{total_files}) - 성공: {processed_count}, 실패: {failed_count}")
        
        success_rate = (processed_count / total_files * 100) if total_files > 0 else 0
        logging.info(f"폴더 '{input_folder}' 처리 완료: {processed_count}개 성공 ({success_rate:.1f}%), {failed_count}개 실패")
        
        self.processed_count += processed_count
        self.failed_count += failed_count
        
        return results
    
    def process_labeled_data(self, data_folder: str) -> List[Dict]:
        """라벨링된 데이터 처리 (normal/abnormal 폴더 구조)"""
        data_path = Path(data_folder)
        all_results = []
        
        # normal 폴더 처리
        normal_folder = data_path / 'normal'
        if normal_folder.exists():
            normal_results = self.process_image_folder(str(normal_folder), 'normal')
            all_results.extend(normal_results)
            logging.info(f"Normal 이미지 {len(normal_results)}개 처리 완료")
        
        # abnormal 폴더 처리
        abnormal_folder = data_path / 'abnormal'
        if abnormal_folder.exists():
            abnormal_results = self.process_image_folder(str(abnormal_folder), 'abnormal')
            all_results.extend(abnormal_results)
            logging.info(f"Abnormal 이미지 {len(abnormal_results)}개 처리 완료")
        
        return all_results
    
    def calculate_statistics(self, data: List[Dict]) -> Dict:
        """통계 계산 (평균, 분산)"""
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        
        # 각도 컬럼들
        angle_columns = ['neck_angle', 'shoulder_angle', 'hip_angle', 'torso_angle']
        
        statistics = {}
        
        # 전체 통계
        for col in angle_columns:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(values) > 0:
                    statistics[f'{col}_mean'] = float(values.mean())
                    statistics[f'{col}_std'] = float(values.std())
                    statistics[f'{col}_var'] = float(values.var())
                    statistics[f'{col}_min'] = float(values.min())
                    statistics[f'{col}_max'] = float(values.max())
        
        # 라벨별 통계
        if 'label' in df.columns:
            for label in df['label'].unique():
                if label == 'unlabeled':
                    continue
                    
                label_data = df[df['label'] == label]
                statistics[f'{label}_count'] = len(label_data)
                
                for col in angle_columns:
                    if col in label_data.columns:
                        values = pd.to_numeric(label_data[col], errors='coerce').dropna()
                        if len(values) > 0:
                            statistics[f'{label}_{col}_mean'] = float(values.mean())
                            statistics[f'{label}_{col}_std'] = float(values.std())
                            statistics[f'{label}_{col}_var'] = float(values.var())
        
        return statistics
    
    def save_data(self, data: List[Dict], output_folder: str = 'data'):
        """데이터 저장"""
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        if not data:
            logging.warning("저장할 데이터가 없습니다.")
            return
        
        # DataFrame 생성
        df = pd.DataFrame(data)
        
        # CSV 저장
        csv_path = output_path / 'pose_data.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logging.info(f"CSV 저장 완료: {csv_path}")
        
        # 통계 계산 및 저장
        stats = self.calculate_statistics(data)
        if stats:
            stats_df = pd.DataFrame([stats])
            stats_path = output_path / 'pose_statistics.csv'
            stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
            logging.info(f"통계 저장 완료: {stats_path}")
            
            # 통계 출력
            logging.info("=== 자세 분석 통계 ===")
            angle_columns = ['neck_angle', 'shoulder_angle', 'hip_angle', 'torso_angle']
            
            for col in angle_columns:
                if f'{col}_mean' in stats:
                    logging.info(f"{col}: 평균={stats[f'{col}_mean']:.2f}°, "
                               f"표준편차={stats[f'{col}_std']:.2f}°, "
                               f"범위=[{stats[f'{col}_min']:.1f}°, {stats[f'{col}_max']:.1f}°]")
            
            # 라벨별 통계
            if 'normal_count' in stats or 'abnormal_count' in stats:
                logging.info("\n=== 라벨별 통계 ===")
                for label in ['normal', 'abnormal']:
                    if f'{label}_count' in stats:
                        logging.info(f"\n{label.upper()} ({stats[f'{label}_count']}개):")
                        for col in angle_columns:
                            if f'{label}_{col}_mean' in stats:
                                logging.info(f"  {col}: 평균={stats[f'{label}_{col}_mean']:.2f}°, "
                                           f"표준편차={stats[f'{label}_{col}_std']:.2f}°")
    
    def __del__(self):
        """소멸자"""
        if hasattr(self, 'pose'):
            self.pose.close()


def main():
    parser = argparse.ArgumentParser(description='자세 데이터 전처리')
    parser.add_argument('--input', default='data/train_images', 
                       help='입력 이미지 폴더 (기본: data/train_images)')
    parser.add_argument('--output', default='data', 
                       help='출력 폴더 (기본: data)')
    parser.add_argument('--labeled', action='store_true', 
                       help='라벨링된 데이터 사용 (normal/abnormal 하위폴더)')
    
    args = parser.parse_args()
    
    # 전처리기 초기화
    preprocessor = PosePreprocessor()
    
    try:
        if args.labeled:
            # 라벨링된 데이터 처리
            logging.info("라벨링된 데이터 처리 시작...")
            all_data = preprocessor.process_labeled_data(args.input)
        else:
            # 단일 폴더 처리
            logging.info("단일 폴더 데이터 처리 시작...")
            all_data = preprocessor.process_image_folder(args.input)
        
        if all_data:
            # 데이터 저장
            preprocessor.save_data(all_data, args.output)
            logging.info(f"전처리 완료: 총 {len(all_data)}개 이미지 처리")
        else:
            logging.warning("처리된 데이터가 없습니다.")
            
    except Exception as e:
        logging.error(f"전처리 중 오류 발생: {e}")
        raise


if __name__ == '__main__':
    main()