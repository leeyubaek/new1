"""
CNN + LSTM 모델 아키텍처 빌더
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, Dropout, 
    BatchNormalization, concatenate, GlobalAveragePooling2D,
    LSTM, TimeDistributed
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


class ModelBuilder:
    """CNN + LSTM 하이브리드 모델 빌더"""
    
    def __init__(self, img_height=112, img_width=112, 
                 sequence_length=5, dropout_rate=0.35):
        self.img_height = img_height
        self.img_width = img_width
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
    
    def _build_cnn_block(self, x, filters, name_prefix):
        """CNN 블록 생성"""
        x = TimeDistributed(Conv2D(
            filters, (3, 3), activation='relu', padding='same', 
            kernel_regularizer=l2(0.001), name=f'{name_prefix}_conv'
        ))(x)
        x = TimeDistributed(BatchNormalization(
            momentum=0.8, name=f'{name_prefix}_bn'
        ))(x)
        x = TimeDistributed(MaxPooling2D(
            (2, 2), name=f'{name_prefix}_pool'
        ))(x)
        x = TimeDistributed(Dropout(
            self.dropout_rate * 0.4, name=f'{name_prefix}_dropout'
        ))(x)
        return x
    
    def _build_lstm_block(self, x, units, return_sequences=True, name_prefix='lstm'):
        """LSTM 블록 생성"""
        x = LSTM(
            units, return_sequences=return_sequences,
            dropout=self.dropout_rate, recurrent_dropout=0.2,
            kernel_regularizer=l2(0.001), name=name_prefix
        )(x)
        x = BatchNormalization(momentum=0.8, name=f'{name_prefix}_bn')(x)
        return x
    
    def build(self, num_numeric_features: int, num_classes: int):
        """CNN + LSTM 하이브리드 모델 구성"""
        
        # 이미지 브랜치
        image_input = Input(
            shape=(self.sequence_length, self.img_height, self.img_width, 3),
            name='image_input'
        )
        
        x = self._build_cnn_block(image_input, 32, 'cnn1')
        x = self._build_cnn_block(x, 64, 'cnn2')
        x = self._build_cnn_block(x, 80, 'cnn3')
        x = TimeDistributed(GlobalAveragePooling2D(name='gap'))(x)
        
        x = self._build_lstm_block(x, 56, True, 'img_lstm1')
        x = self._build_lstm_block(x, 28, False, 'img_lstm2')
        image_features = Dense(28, activation='relu', 
                              kernel_regularizer=l2(0.0007), 
                              name='image_features')(x)
        
        # 수치 브랜치
        numeric_input = Input(
            shape=(self.sequence_length, num_numeric_features),
            name='numeric_input'
        )
        
        y = self._build_lstm_block(numeric_input, 28, True, 'num_lstm1')
        y = self._build_lstm_block(y, 14, False, 'num_lstm2')
        numeric_features = Dense(14, activation='relu', 
                                kernel_regularizer=l2(0.0007),
                                name='numeric_features')(y)
        
        # 융합 브랜치
        merged = concatenate([image_features, numeric_features], name='fusion')
        
        z = Dense(28, activation='relu', kernel_regularizer=l2(0.0007))(merged)
        z = BatchNormalization(momentum=0.8)(z)
        z = Dropout(self.dropout_rate)(z)
        z = Dense(14, activation='relu', kernel_regularizer=l2(0.0007))(z)
        z = Dropout(self.dropout_rate * 0.8)(z)
        
        output = Dense(num_classes, activation='softmax', name='output')(z)
        
        # 모델 생성 및 컴파일
        model = Model(
            inputs=[image_input, numeric_input],
            outputs=output,
            name='CNN_LSTM_Posture_Model'
        )
        
        # Mixed Precision을 위한 Loss Scaling
        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        
        # Mixed Precision 사용 시 Loss Scaling 적용
        try:
            from tensorflow.keras import mixed_precision
            if mixed_precision.global_policy().name == 'mixed_float16':
                optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        except:
            pass
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        
        return model
