"""
CNN + LSTM 모델 유틸리티 함수들
- 데이터 로드, 증강, 시퀀스 생성 등
"""

import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

def load_images_from_folder(image_dir: str, labels: np.ndarray, 
                            img_width: int, img_height: int) -> Tuple:
    """폴더에서 이미지 로드 (한글 경로 지원 + 병렬 처리)"""
    image_path = Path(image_dir)
    
    logging.info(f"이미지 로드 시작: {len(labels)}개 파일 (병렬 처리)")
    
    def load_single_image(idx, label):
        """단일 이미지 로드 함수 (병렬 처리용)"""
        label_folder = image_path / label
        
        if not label_folder.exists():
            return None, idx, "folder_not_found"
        
        # 이미지 파일 찾기
        img_files = (
            list(label_folder.glob('*.jpg')) + list(label_folder.glob('*.JPG')) +
            list(label_folder.glob('*.png')) + list(label_folder.glob('*.PNG')) +
            list(label_folder.glob('*.jpeg')) + list(label_folder.glob('*.JPEG'))
        )
        
        if not img_files:
            return None, idx, "no_images"
        
        img_idx = idx % len(img_files)
        img_path = img_files[img_idx]
        
        try:
            # 한글 경로 처리
            img_array = np.fromfile(str(img_path), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                img = cv2.imread(str(img_path))
            
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)
                return img, idx, "success"
            else:
                return None, idx, "decode_failed"
                
        except Exception as e:
            return None, idx, f"error: {str(e)[:30]}"
    
    # 병렬 처리로 이미지 로드
    max_workers = min(8, multiprocessing.cpu_count())
    results = {}
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_single_image, idx, label): idx 
                  for idx, label in enumerate(labels)}
        
        completed = 0
        total = len(futures)
        
        for future in as_completed(futures):
            completed += 1
            img, idx, status = future.result()
            
            if status == "success":
                results[idx] = img
            else:
                failed_count += 1
            
            if completed % max(1, total // 10) == 0:
                progress = completed / total * 100
                success_count = len(results)
                logging.info(f"진행률: {progress:.0f}% ({success_count}/{completed})")
    
    # 인덱스 순서대로 정렬
    images = []
    valid_indices = []
    for idx in sorted(results.keys()):
        images.append(results[idx])
        valid_indices.append(idx)
    
    if len(images) == 0:
        raise ValueError(
            f"이미지를 로드할 수 없습니다. 폴더를 확인하세요: {image_dir}\n"
            f"필요한 폴더: {set(labels)}"
        )
    
    images = np.array(images, dtype=np.float32) / 255.0
    
    success_rate = len(images) / len(labels) * 100
    logging.info(
        f"이미지 로드 완료: {len(images)}개 성공 ({success_rate:.1f}%), "
        f"{failed_count}개 실패"
    )
    
    return images, valid_indices


def create_sequences(images: np.ndarray, numeric_data: np.ndarray, 
                    labels: np.ndarray, sequence_length: int) -> Tuple:
    """시퀀스 데이터 생성"""
    img_sequences = []
    num_sequences = []
    seq_labels = []
    
    unique_labels = np.unique(labels)
    
    logging.info(f"시퀀스 생성 시작 (sequence_length={sequence_length})")
    
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        
        if len(label_indices) < sequence_length:
            logging.warning(
                f"라벨 '{label}'의 데이터가 부족합니다: "
                f"{len(label_indices)}개 < {sequence_length}개"
            )
            continue
        
        for i in range(len(label_indices) - sequence_length + 1):
            indices = label_indices[i:i + sequence_length]
            
            try:
                img_sequences.append(images[indices])
                num_sequences.append(numeric_data[indices])
                seq_labels.append(label)
            except IndexError as e:
                logging.error(f"인덱스 오류: {e}")
                continue
    
    if len(img_sequences) == 0:
        raise ValueError(
            f"시퀀스를 생성할 수 없습니다. "
            f"각 라벨당 최소 {sequence_length}개의 데이터가 필요합니다."
        )
    
    img_sequences = np.array(img_sequences)
    num_sequences = np.array(num_sequences)
    seq_labels = np.array(seq_labels)
    
    logging.info(
        f"시퀀스 생성 완료: {len(img_sequences)}개 "
        f"(라벨: {dict(zip(*np.unique(seq_labels, return_counts=True)))})"
    )
    
    return img_sequences, num_sequences, seq_labels


def augment_data(img_seqs: np.ndarray, num_seqs: np.ndarray, 
                labels: np.ndarray, augment_factor: int = 1, 
                target_class: int = None) -> Tuple:
    """데이터 증강"""
    aug_img_seqs = []
    aug_num_seqs = []
    aug_labels = []
    
    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        fill_mode='nearest',
        horizontal_flip=False
    )
    
    target_msg = f"클래스 {target_class}" if target_class is not None else "모든 클래스"
    logging.info(f"데이터 증강 시작 ({target_msg}, factor={augment_factor})")
    
    for i in range(len(img_seqs)):
        current_label = labels[i]
        
        # 원본 항상 추가
        aug_img_seqs.append(img_seqs[i])
        aug_num_seqs.append(num_seqs[i])
        aug_labels.append(labels[i])
        
        if target_class is not None and current_label != target_class:
            continue
        
        # 증강 데이터 생성
        for _ in range(augment_factor):
            aug_img_seq = []
            
            for frame in img_seqs[i]:
                img = frame.reshape((1,) + frame.shape)
                aug_img = datagen.flow(img, batch_size=1)[0][0]
                aug_img = np.clip(aug_img, 0, 1)
                aug_img_seq.append(aug_img)
            
            aug_img_seqs.append(np.array(aug_img_seq))
            
            noise = np.random.normal(0, 0.005, num_seqs[i].shape)
            aug_num_data = num_seqs[i] + noise
            aug_num_data = np.clip(aug_num_data, -180, 180)
            aug_num_seqs.append(aug_num_data)
            aug_labels.append(labels[i])
        
        if (i + 1) % max(1, len(img_seqs) // 10) == 0:
            progress = (i + 1) / len(img_seqs) * 100
            logging.info(f"증강 진행률: {progress:.0f}%")
    
    aug_img_seqs = np.array(aug_img_seqs)
    aug_num_seqs = np.array(aug_num_seqs)
    aug_labels = np.array(aug_labels)
    
    unique, counts = np.unique(aug_labels, return_counts=True)
    class_dist = dict(zip(unique, counts))
    logging.info(f"데이터 증강 완료: {len(aug_img_seqs)}개 (클래스 분포: {class_dist})")
    
    return aug_img_seqs, aug_num_seqs, aug_labels


def split_and_augment_data(X_img, X_num, y, augment: bool):
    """데이터 분할 및 증강"""
    # Train/Test 분할 (85/15)
    X_img_temp, X_img_test, X_num_temp, X_num_test, y_temp, y_test = train_test_split(
        X_img, X_num, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Train/Val 분할 (70/15)
    X_img_train, X_img_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
        X_img_temp, X_num_temp, y_temp, test_size=0.18, random_state=42, stratify=y_temp
    )
    
    original_train_count = len(X_img_train)
    original_train_class_dist = dict(zip(*np.unique(y_train, return_counts=True)))
    
    logging.info(f"="*60)
    logging.info(f"원본 데이터 분할 - 훈련: {len(X_img_train)}, 검증: {len(X_img_val)}, 테스트: {len(X_img_test)}")
    logging.info(f"훈련 세트 클래스 분포: {original_train_class_dist}")
    logging.info(f"="*60)
    
    # 훈련 데이터 증강
    if augment and len(X_img_train) > 0:
        unique, counts = np.unique(y_train, return_counts=True)
        if len(unique) > 0:
            aug_factor = 1
            logging.info(f"모든 클래스를 {aug_factor}배 증강합니다.")
            
            X_img_train, X_num_train, y_train = augment_data(
                X_img_train, X_num_train, y_train, 
                augment_factor=aug_factor,
                target_class=None
            )
    
    augmented_train_count = len(X_img_train)
    
    if augment and augmented_train_count > original_train_count:
        increase_count = augmented_train_count - original_train_count
        increase_percent = (increase_count / original_train_count) * 100
        
        logging.info(f"="*60)
        logging.info(f"✅ 데이터 증강 완료!")
        logging.info(f"원본 훈련 데이터: {original_train_count}개")
        logging.info(f"증강 후 훈련 데이터: {augmented_train_count}개")
        logging.info(f"증가량: +{increase_count}개 ({increase_percent:.1f}% 증가)")
        logging.info(f"="*60)
    
    total = len(X_img_train) + len(X_img_val) + len(X_img_test)
    logging.info(
        f"최종 분포 - 훈련: {len(X_img_train)} ({len(X_img_train)/total*100:.1f}%), "
        f"검증: {len(X_img_val)} ({len(X_img_val)/total*100:.1f}%), "
        f"테스트: {len(X_img_test)} ({len(X_img_test)/total*100:.1f}%)"
    )
    
    return (X_img_train, X_num_train, y_train,
            X_img_val, X_num_val, y_val,
            X_img_test, X_num_test, y_test)
