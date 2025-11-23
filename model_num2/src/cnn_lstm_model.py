"""
CNN + LSTM 하이브리드 자세 평가 모델
- 이미지 특징 추출 (CNN) + 시계열 패턴 학습 (LSTM)
- 정상/비정상 자세 분류
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, Dropout, 
    BatchNormalization, concatenate, GlobalAveragePooling2D,
    LSTM, TimeDistributed
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import joblib
import argparse
import cv2
from typing import Tuple, List

# 로깅 설정
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

class CNNLSTMModel:
    """CNN + LSTM 하이브리드 자세 분류 모델"""
    
    def __init__(self, img_height=128, img_width=128, sequence_length=5, dropout_rate=0.4):
        """
        Args:
            img_height (int): 이미지 높이 (default: 128)
            img_width (int): 이미지 너비 (default: 128)
            sequence_length (int): LSTM 시퀀스 길이 (default: 5)
            dropout_rate (float): 드롭아웃 비율 (default: 0.4)
        """
        self.img_height = img_height
        self.img_width = img_width
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        
    def load_data(self, csv_path: str, image_dir: str) -> Tuple:
        """CSV 데이터 및 이미지 경로 로드"""
        try:
            df = pd.read_csv(csv_path)
            logging.info(f"데이터 로드 완료: {len(df)}개 샘플")
            
            # 필수 컬럼 확인
            required_columns = ['neck_angle', 'shoulder_angle', 'hip_angle', 'torso_angle', 'label']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
            
            # 특성 컬럼 정의
            self.feature_columns = [
                'neck_angle', 'shoulder_angle', 'hip_angle', 'torso_angle'
            ]
            
            # 좌표 특성 추가
            coordinate_features = [
                f'{part}_{coord}'
                for part in ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
                for coord in ['x', 'y']
                if f'{part}_{coord}' in df.columns
            ]
            
            if coordinate_features:
                self.feature_columns.extend(coordinate_features)
                logging.info(f"좌표 특성 {len(coordinate_features)}개 추가")
            
            # 결측값 처리
            df = df.dropna(subset=self.feature_columns + ['label'])
            logging.info(f"결측값 제거 후: {len(df)}개 샘플")
            
            # 'unlabeled' 제거
            if 'unlabeled' in df['label'].values:
                df = df[df['label'] != 'unlabeled']
                logging.info(f"unlabeled 제거 후: {len(df)}개 샘플")
            
            # 라벨 분포 확인
            label_counts = df['label'].value_counts()
            logging.info(f"라벨 분포: {dict(label_counts)}")
            
            return df, image_dir
            
        except Exception as e:
            logging.error(f"데이터 로드 실패: {e}")
            raise
    
    def load_images_from_folder(self, image_dir: str, labels: np.ndarray) -> Tuple:
        """폴더에서 이미지 로드"""
        image_path = Path(image_dir)
        images = []
        valid_indices = []
        
        for idx, label in enumerate(labels):
            label_folder = image_path / label
            if not label_folder.exists():
                logging.warning(f"폴더가 없습니다: {label_folder}")
                continue
            
            img_files = list(label_folder.glob('*.jpg')) + \
                       list(label_folder.glob('*.png')) + \
                       list(label_folder.glob('*.jpeg'))
            
            if img_files:
                img_path = img_files[min(idx % len(img_files), len(img_files)-1)]
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.img_width, self.img_height))
                    images.append(img)
                    valid_indices.append(idx)
        
        if len(images) == 0:
            raise ValueError("이미지를 로드할 수 없습니다. 이미지 폴더를 확인하세요.")
        
        images = np.array(images, dtype=np.float32) / 255.0
        logging.info(f"이미지 로드 완료: {len(images)}개")
        
        return images, valid_indices
    
    def create_sequences(self, images: np.ndarray, numeric_data: np.ndarray, 
                        labels: np.ndarray) -> Tuple:
        """시퀀스 데이터 생성"""
        img_sequences = []
        num_sequences = []
        seq_labels = []
        
        # 라벨별로 시퀀스 생성
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            
            for i in range(len(label_indices) - self.sequence_length + 1):
                indices = label_indices[i:i + self.sequence_length]
                img_sequences.append(images[indices])
                num_sequences.append(numeric_data[indices])
                seq_labels.append(label)
        
        img_sequences = np.array(img_sequences)
        num_sequences = np.array(num_sequences)
        seq_labels = np.array(seq_labels)
        
        logging.info(f"시퀀스 생성 완료: {len(img_sequences)}개")
        
        return img_sequences, num_sequences, seq_labels
    
    def augment_data(self, img_seqs: np.ndarray, num_seqs: np.ndarray, 
                    labels: np.ndarray, augment_factor: int = 1) -> Tuple:
        """
        데이터 증강
        
        Args:
            img_seqs: 이미지 시퀀스 배열
            num_seqs: 수치 시퀀스 배열
            labels: 라벨 배열
            augment_factor: 증강 팩터 (default: 1)
            
        Returns:
            증강된 이미지, 수치, 라벨 배열
        """
        aug_img_seqs = []
        aug_num_seqs = []
        aug_labels = []
        
        datagen = ImageDataGenerator(
            rotation_range=5,  # 8 -> 5로 감소
            width_shift_range=0.05,  # 0.08 -> 0.05로 감소
            height_shift_range=0.05,  # 0.08 -> 0.05로 감소
            zoom_range=0.05,  # 0.08 -> 0.05로 감소
            fill_mode='nearest'
        )
        
        for i in range(len(img_seqs)):
            # 원본
            aug_img_seqs.append(img_seqs[i])
            aug_num_seqs.append(num_seqs[i])
            aug_labels.append(labels[i])
            
            # 증강
            for _ in range(augment_factor):
                aug_img_seq = []
                for frame in img_seqs[i]:
                    img = frame.reshape((1,) + frame.shape)
                    aug_img = datagen.flow(img, batch_size=1)[0][0]
                    aug_img_seq.append(aug_img)
                
                aug_img_seqs.append(np.array(aug_img_seq))
                
                # 노이즈 감소 (0.01 -> 0.005)
                noise = np.random.normal(0, 0.005, num_seqs[i].shape)
                aug_num_seqs.append(num_seqs[i] + noise)
                aug_labels.append(labels[i])
        
        aug_img_seqs = np.array(aug_img_seqs)
        aug_num_seqs = np.array(aug_num_seqs)
        aug_labels = np.array(aug_labels)
        
        logging.info(f"데이터 증강 완료: {len(aug_img_seqs)}개")
        
        return aug_img_seqs, aug_num_seqs, aug_labels
    
    def prepare_data(self, df: pd.DataFrame, image_dir: str, 
                    augment: bool = True) -> Tuple:
        """데이터 전처리 및 준비"""
        X_numeric = df[self.feature_columns].values
        y = df['label'].values
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        X_images, valid_indices = self.load_images_from_folder(image_dir, y)
        
        X_numeric = X_numeric[valid_indices]
        y_encoded = y_encoded[valid_indices]
        
        self.scaler = StandardScaler()
        X_numeric_scaled = self.scaler.fit_transform(X_numeric)
        
        # 시퀀스 생성
        X_img_seq, X_num_seq, y_seq = self.create_sequences(
            X_images, X_numeric_scaled, y_encoded
        )
        
        if len(X_img_seq) == 0:
            raise ValueError("시퀀스를 생성할 수 없습니다.")
        
        # 데이터 분할 및 증강
        return self._split_and_augment_data(
            X_img_seq, X_num_seq, y_seq, augment
        )
    
    def _split_and_augment_data(self, X_img, X_num, y, augment):
        """데이터 분할 및 증강 헬퍼 메서드"""
        # Train/Temp 분할 (85/15)
        X_img_temp, X_img_test, X_num_temp, X_num_test, y_temp, y_test = train_test_split(
            X_img, X_num, y, test_size=0.15, random_state=42, stratify=y
        )
        
        # Train/Val 분할 (70/15 from temp)
        X_img_train, X_img_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
            X_img_temp, X_num_temp, y_temp, test_size=0.18, random_state=42, stratify=y_temp
        )
        
        logging.info(f"분할 - 훈련: {len(X_img_train)}, 검증: {len(X_img_val)}, 테스트: {len(X_img_test)}")
        
        # 훈련 데이터 증강
        if augment and len(X_img_train) > 0:
            X_img_train, X_num_train, y_train = self.augment_data(
                X_img_train, X_num_train, y_train, augment_factor=1
            )
        
        # 최종 분포 출력
        total = len(X_img_train) + len(X_img_val) + len(X_img_test)
        logging.info(
            f"최종 분포 - 훈련: {len(X_img_train)} ({len(X_img_train)/total*100:.1f}%), "
            f"검증: {len(X_img_val)} ({len(X_img_val)/total*100:.1f}%), "
            f"테스트: {len(X_img_test)} ({len(X_img_test)/total*100:.1f}%)"
        )
        
        return (X_img_train, X_num_train, y_train,
                X_img_val, X_num_val, y_val,
                X_img_test, X_num_test, y_test)
    
    def _build_cnn_block(self, x, filters, name_prefix):
        """
CNN 블록 생성 헬퍼
        
        Args:
            x: 입력 텔서
            filters: 필터 수
            name_prefix: 레이어 이름 접두사
            
        Returns:
            처리된 텔서
        """
        x = TimeDistributed(Conv2D(
            filters, (3, 3), activation='relu', padding='same', 
            kernel_regularizer=l2(0.001), name=f'{name_prefix}_conv'
        ))(x)
        x = TimeDistributed(BatchNormalization(momentum=0.9, name=f'{name_prefix}_bn'))(x)
        x = TimeDistributed(MaxPooling2D((2, 2), name=f'{name_prefix}_pool'))(x)
        x = TimeDistributed(Dropout(self.dropout_rate * 0.5, name=f'{name_prefix}_dropout'))(x)
        return x
    
    def _build_lstm_block(self, x, units, return_sequences=True, name_prefix='lstm'):
        """
LSTM 블록 생성 헬퍼
        
        Args:
            x: 입력 텔서
            units: LSTM 유닛 수
            return_sequences: 시퀀스 반환 여부
            name_prefix: 레이어 이름 접두사
            
        Returns:
            처리된 텔서
        """
        x = LSTM(
            units, return_sequences=return_sequences,
            dropout=self.dropout_rate, recurrent_dropout=0.2,
            kernel_regularizer=l2(0.001), name=name_prefix
        )(x)
        x = BatchNormalization(momentum=0.9, name=f'{name_prefix}_bn')(x)
        return x
    
    def build_model(self, num_numeric_features: int, num_classes: int):
        """CNN + LSTM 하이브리드 모델 구성"""
        
        # 이미지 브랜치
        image_input = Input(
            shape=(self.sequence_length, self.img_height, self.img_width, 3),
            name='image_input'
        )
        
        # CNN 블록
        x = self._build_cnn_block(image_input, 32, 'cnn1')
        x = self._build_cnn_block(x, 64, 'cnn2')
        x = self._build_cnn_block(x, 128, 'cnn3')
        x = TimeDistributed(GlobalAveragePooling2D(name='gap'))(x)
        
        # LSTM 블록
        x = self._build_lstm_block(x, 64, True, 'img_lstm1')
        x = self._build_lstm_block(x, 32, False, 'img_lstm2')
        image_features = Dense(32, activation='relu', kernel_regularizer=l2(0.001), 
                              name='image_features')(x)
        
        # 수치 브랜치
        numeric_input = Input(
            shape=(self.sequence_length, num_numeric_features),
            name='numeric_input'
        )
        
        y = self._build_lstm_block(numeric_input, 32, True, 'num_lstm1')
        y = self._build_lstm_block(y, 16, False, 'num_lstm2')
        numeric_features = Dense(16, activation='relu', kernel_regularizer=l2(0.001),
                                name='numeric_features')(y)
        
        # 융합 브랜치
        merged = concatenate([image_features, numeric_features], name='fusion')
        
        z = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(merged)
        z = BatchNormalization(momentum=0.9)(z)
        z = Dropout(self.dropout_rate)(z)
        z = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(z)
        z = Dropout(self.dropout_rate)(z)
        
        output = Dense(num_classes, activation='softmax', name='output')(z)
        
        # 모델 생성 및 컴파일
        self.model = Model(
            inputs=[image_input, numeric_input],
            outputs=output,
            name='CNN_LSTM_Posture_Model'
        )
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model.summary()
        
        return self.model
    
    def train(self, X_img_train, X_num_train, y_train,
              X_img_val, X_num_val, y_val,
              epochs=100, batch_size=8):
        """
        모델 훈련
        
        Args:
            X_img_train, X_num_train, y_train: 훈련 데이터
            X_img_val, X_num_val, y_val: 검증 데이터
            epochs: 훈련 에폭 수
            batch_size: 배치 크기
            
        Returns:
            훈련 히스토리
        """
        if self.model is None:
            raise ValueError("모델이 구성되지 않았습니다.")
        
        # 클래스 가중치 계산 (불균형 해결)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        logging.info(f"클래스 가중치: {class_weight_dict}")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=25,  # patience 증가 (안정성 향상)
                restore_best_weights=True,
                min_delta=0.001,  # 최소 개선 임계값 추가
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.6,  # 학습률 감소 속도 완화
                patience=10,  # patience 증가
                min_lr=1e-7,
                min_delta=0.001,  # 최소 개선 임계값 추가
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_cnn_lstm_model.h5',
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            )
        ]
        
        history = self.model.fit(
            [X_img_train, X_num_train], y_train,
            validation_data=([X_img_val, X_num_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,  # 클래스 가중치 적용
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_img, X_num, y, dataset_name="Test"):
        """
        모델 평가 및 상세 메트릭 계산
        
        Args:
            X_img: 이미지 데이터
            X_num: 수치 데이터
            y: 실제 라벨
            dataset_name: 데이터셋 이름 (Train/Validation/Test)
            
        Returns:
            accuracy, report, confusion_matrix, metrics
        """
        if self.model is None:
            raise ValueError("훈련된 모델이 없습니다.")
        
        # 예측
        y_pred_proba = self.model.predict([X_img, X_num], verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 기본 메트릭
        accuracy = accuracy_score(y, y_pred)
        
        # 클래스별 메트릭 계산
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        # 결과 출력
        logging.info(f"\n{'='*60}")
        logging.info(f"{dataset_name} 세트 평가 결과")
        logging.info(f"{'='*60}")
        logging.info(f"정확도 (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)")
        logging.info(f"정밀도 (Precision): {precision:.4f} ({precision*100:.2f}%)")
        logging.info(f"재현율 (Recall):    {recall:.4f} ({recall*100:.2f}%)")
        logging.info(f"F1 점수:          {f1:.4f} ({f1*100:.2f}%)")
        
        # 클래스별 성능
        class_names = self.label_encoder.classes_
        report = classification_report(y, y_pred, target_names=class_names, zero_division=0)
        logging.info(f"\n{dataset_name} 분류 보고서:\n{report}")
        
        # 혼동행렬
        cm = confusion_matrix(y, y_pred)
        
        # 예측 신뢰도 분석
        avg_confidence = np.mean(np.max(y_pred_proba, axis=1))
        logging.info(f"평균 예측 신뢰도: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
        
        # 혼동행렬 시각화
        self._plot_confusion_matrix(cm, class_names, accuracy, dataset_name)
        
        # 클래스별 성능 시각화
        self._plot_class_performance(y, y_pred, y_pred_proba, class_names, dataset_name)
        
        # 종합 메트릭 리턴
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_confidence': avg_confidence,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        return accuracy, report, cm, metrics
    
    def _plot_confusion_matrix(self, cm, class_names, accuracy, dataset_name):
        """혼동행렬 시각화"""
        plt.figure(figsize=(10, 8))
        
        # 개수만 표시 (퍼센트 제거)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'},
                   linewidths=1, linecolor='gray')
        
        plt.title(f'{dataset_name} Confusion Matrix\nAccuracy: {accuracy:.4f}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # 파일 저장
        save_path = f'models/confusion_matrix_{dataset_name.lower()}_cnn_lstm.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"혼동행렬 저장 완료: {save_path}")
        
        plt.show()
        plt.close()
    
    def _plot_class_performance(self, y_true, y_pred, y_pred_proba, class_names, dataset_name):
        """클래스별 성능 시각화"""
        from sklearn.metrics import precision_recall_fscore_support
        
        # 클래스별 메트릭 계산
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(class_names)), zero_division=0
        )
        
        # 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. 클래스별 메트릭 비교
        x = np.arange(len(class_names))
        width = 0.25
        
        ax1.bar(x - width, precision, width, label='Precision', alpha=0.8, color='#2ecc71')
        ax1.bar(x, recall, width, label='Recall', alpha=0.8, color='#3498db')
        ax1.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='#e74c3c')
        
        ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title(f'{dataset_name} Class Performance Metrics', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_names, rotation=45, ha='right')
        ax1.legend(fontsize=10)
        ax1.set_ylim([0, 1.1])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 값 표시
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            ax1.text(i - width, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=8)
            ax1.text(i, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=8)
            ax1.text(i + width, f + 0.02, f'{f:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 2. 클래스별 샘플 수 및 예측 신뢰도
        avg_confidence_per_class = []
        for i in range(len(class_names)):
            mask = y_true == i
            if mask.sum() > 0:
                avg_conf = np.mean(np.max(y_pred_proba[mask], axis=1))
                avg_confidence_per_class.append(avg_conf)
            else:
                avg_confidence_per_class.append(0)
        
        ax2_twin = ax2.twinx()
        
        bars = ax2.bar(x, support, alpha=0.6, color='#9b59b6', label='Sample Count')
        line = ax2_twin.plot(x, avg_confidence_per_class, 'o-', color='#e67e22', 
                            linewidth=2, markersize=8, label='Avg Confidence')
        
        ax2.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Sample Count', fontsize=12, fontweight='bold', color='#9b59b6')
        ax2_twin.set_ylabel('Average Confidence', fontsize=12, fontweight='bold', color='#e67e22')
        ax2.set_title(f'{dataset_name} Sample Distribution & Confidence', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
        ax2.tick_params(axis='y', labelcolor='#9b59b6')
        ax2_twin.tick_params(axis='y', labelcolor='#e67e22')
        ax2_twin.set_ylim([0, 1.1])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 레전드 통합
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        
        # 값 표시
        for i, (s, c) in enumerate(zip(support, avg_confidence_per_class)):
            ax2.text(i, s + max(support) * 0.02, str(s), ha='center', va='bottom', fontsize=9)
            ax2_twin.text(i, c + 0.03, f'{c:.2f}', ha='center', va='bottom', fontsize=8, color='#e67e22')
        
        plt.tight_layout()
        
        # 저장
        save_path = f'models/class_performance_{dataset_name.lower()}_cnn_lstm.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"클래스 성능 분서 저장 완료: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_training_history(self, history):
        """
        훈련 히스토리 시각화
        
        Args:
            history: 모델 훈련 히스토리
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 에포크 수
        epochs = range(1, len(history.history['accuracy']) + 1)
        
        # 정확도 그래프
        ax1.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=3)
        ax1.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=3)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(fontsize=10, loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.0])
        
        # 손실률 그래프
        ax2.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2, marker='o', markersize=3)
        ax2.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=3)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(fontsize=10, loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/training_history_cnn_lstm.png', dpi=300, bbox_inches='tight')
        logging.info("훈련 히스토리 그래프 저장 완료: models/training_history_cnn_lstm.png")
        plt.show()
        plt.close()
    
    def save_model(self, model_path: str = 'models/cnn_lstm_model.h5'):
        """
        모델 및 전처리 객체 저장
        
        Args:
            model_path: 모델 저장 경로
        """
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        self.model.save(model_path)
        joblib.dump(self.scaler, 'models/scaler_cnn_lstm.pkl')
        joblib.dump(self.label_encoder, 'models/label_encoder_cnn_lstm.pkl')
        
        metadata = {
            'img_height': self.img_height,
            'img_width': self.img_width,
            'sequence_length': self.sequence_length,
            'dropout_rate': self.dropout_rate,
            'feature_columns': self.feature_columns,
            'classes': self.label_encoder.classes_.tolist()
        }
        
        import json
        with open('models/model_metadata_cnn_lstm.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"모델 저장 완료: {model_path}")


def main():
    parser = argparse.ArgumentParser(description='CNN + LSTM 자세 분류 모델 훈련')
    parser.add_argument('--data', default='data/pose_data.csv',
                       help='훈련 데이터 CSV 파일 경로')
    parser.add_argument('--images', default='data/train_images',
                       help='이미지 폴더 경로')
    parser.add_argument('--epochs', type=int, default=50,
                       help='훈련 에폭 수')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='배치 크기')
    parser.add_argument('--img_size', type=int, default=128,
                       help='이미지 크기 (정사각형)')
    parser.add_argument('--sequence_length', type=int, default=5,
                       help='LSTM 시퀀스 길이')
    parser.add_argument('--no_augment', action='store_true',
                       help='데이터 증강 사용 안 함')
    
    args = parser.parse_args()
    
    Path('models').mkdir(exist_ok=True)
    
    model = CNNLSTMModel(
        img_height=args.img_size,
        img_width=args.img_size,
        sequence_length=args.sequence_length
    )
    
    try:
        logging.info("데이터 로드 중...")
        df, image_dir = model.load_data(args.data, args.images)
        
        logging.info("데이터 전처리 중...")
        (X_img_train, X_num_train, y_train,
         X_img_val, X_num_val, y_val,
         X_img_test, X_num_test, y_test) = model.prepare_data(
            df, image_dir, augment=not args.no_augment
        )
        
        if len(X_img_train) == 0:
            logging.error("훈련 데이터가 없습니다.")
            return
        
        logging.info("모델 구성 중...")
        num_classes = len(model.label_encoder.classes_)
        num_numeric_features = len(model.feature_columns)
        
        model.build_model(num_numeric_features, num_classes)
        
        logging.info("모델 훈련 시작...")
        history = model.train(
            X_img_train, X_num_train, y_train,
            X_img_val, X_num_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        model.plot_training_history(history)
        
        logging.info("훈련 세트 평가 중...")
        train_accuracy, train_report, train_cm, train_metrics = model.evaluate(
            X_img_train, X_num_train, y_train, dataset_name="Train"
        )
        
        logging.info("검증 세트 평가 중...")
        val_accuracy, val_report, val_cm, val_metrics = model.evaluate(
            X_img_val, X_num_val, y_val, dataset_name="Validation"
        )
        
        logging.info("테스트 세트 평가 중...")
        test_accuracy, test_report, test_cm, test_metrics = model.evaluate(
            X_img_test, X_num_test, y_test, dataset_name="Test"
        )
        
        logging.info("모델 저장 중...")
        model.save_model()
        
        # 최종 결과 요약
        logging.info("\n" + "="*70)
        logging.info("CNN + LSTM 모델 훈련 완료!")
        logging.info("="*70)
        
        logging.info("\n[훈련 세트 결과]")
        logging.info(f"  정확도: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        logging.info(f"  Precision: {train_metrics['precision']:.4f}")
        logging.info(f"  Recall: {train_metrics['recall']:.4f}")
        logging.info(f"  F1-Score: {train_metrics['f1_score']:.4f}")
        
        logging.info("\n[검증 세트 결과]")
        logging.info(f"  정확도: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        logging.info(f"  Precision: {val_metrics['precision']:.4f}")
        logging.info(f"  Recall: {val_metrics['recall']:.4f}")
        logging.info(f"  F1-Score: {val_metrics['f1_score']:.4f}")
        
        logging.info("\n[테스트 세트 결과] ⭐")
        logging.info(f"  정확도: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        logging.info(f"  Precision: {test_metrics['precision']:.4f}")
        logging.info(f"  Recall: {test_metrics['recall']:.4f}")
        logging.info(f"  F1-Score: {test_metrics['f1_score']:.4f}")
        logging.info(f"  평균 신뢰도: {test_metrics['avg_confidence']:.4f}")
        
        # 과적합 검사
        overfitting_gap = train_accuracy - val_accuracy
        if overfitting_gap > 0.1:
            logging.warning(f"\n⚠️  과적합 가능성: 훈련-검증 정확도 차이 {overfitting_gap:.4f}")
        else:
            logging.info(f"\n✅ 좋은 일반화 성능: 훈련-검증 정확도 차이 {overfitting_gap:.4f}")
        
        logging.info("\n" + "="*70)
        
    except Exception as e:
        logging.error(f"훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
