"""
모델 평가 및 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)


def evaluate_model(model, X_img, X_num, y, label_encoder, dataset_name="Test"):
    """모델 평가 및 상세 메트릭 계산"""
    
    # 예측
    y_pred_proba = model.predict([X_img, X_num], verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 메트릭 계산
    from sklearn.metrics import precision_score, recall_score, f1_score
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
    
    # 결과 출력
    logging.info(f"\n{'='*60}")
    logging.info(f"{dataset_name} 세트 평가 결과")
    logging.info(f"{'='*60}")
    logging.info(f"정확도:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    logging.info(f"정밀도: {precision:.4f} ({precision*100:.2f}%)")
    logging.info(f"재현율: {recall:.4f} ({recall*100:.2f}%)")
    logging.info(f"F1 점수: {f1:.4f} ({f1*100:.2f}%)")
    
    # 클래스별 성능
    class_names = label_encoder.classes_
    
    # 클래스 순서 변경: normal을 먼저
    if len(class_names) == 2 and class_names[0] == 'abnormal':
        class_names_reordered = [class_names[1], class_names[0]]
        class_indices_reordered = [1, 0]
    else:
        class_names_reordered = class_names
        class_indices_reordered = list(range(len(class_names)))
    
    report = classification_report(y, y_pred, target_names=class_names, zero_division=0)
    logging.info(f"\n{dataset_name} 분류 보고서:\n{report}")
    
    # 혼동행렬
    cm = confusion_matrix(y, y_pred)
    if len(class_names) == 2 and class_names[0] == 'abnormal':
        cm_reordered = cm[np.ix_(class_indices_reordered, class_indices_reordered)]
    else:
        cm_reordered = cm
    
    avg_confidence = np.mean(np.max(y_pred_proba, axis=1))
    logging.info(f"평균 예측 신뢰도: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
    
    # 시각화
    plot_confusion_matrix(cm_reordered, class_names_reordered, accuracy, dataset_name)
    plot_class_performance(y, y_pred, y_pred_proba, class_names_reordered, 
                          class_indices_reordered, dataset_name)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'avg_confidence': avg_confidence,
        'confusion_matrix': cm_reordered,
        'classification_report': report
    }
    
    return accuracy, report, cm, metrics


def plot_confusion_matrix(cm, class_names, accuracy, dataset_name):
    """혼동행렬 시각화"""
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Count'},
               linewidths=1, linecolor='gray')
    
    plt.title(f'{dataset_name} Confusion Matrix\nAccuracy: {accuracy:.4f}', 
             fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    save_path = f'models/confusion_matrix_{dataset_name.lower()}_cnn_lstm.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"혼동행렬 저장: {save_path}")
    
    plt.show()
    plt.close()


def plot_class_performance(y_true, y_pred, y_pred_proba, class_names, 
                          class_indices, dataset_name):
    """클래스별 성능 시각화"""
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_indices, zero_division=0
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 클래스별 메트릭 비교
    x = np.arange(len(class_names))
    width = 0.25
    
    ax1.bar(x - width, precision, width, label='Precision', alpha=0.8, color='#2ecc71')
    ax1.bar(x, recall, width, label='Recall', alpha=0.8, color='#3498db')
    ax1.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='#e74c3c')
    
    ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title(f'{dataset_name} Class Performance', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax1.text(i - width, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width, f + 0.02, f'{f:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 2. 샘플 수 및 신뢰도
    avg_confidence_per_class = []
    for orig_idx in class_indices:
        mask = y_true == orig_idx
        if mask.sum() > 0:
            avg_conf = np.mean(np.max(y_pred_proba[mask], axis=1))
            avg_confidence_per_class.append(avg_conf)
        else:
            avg_confidence_per_class.append(0)
    
    ax2_twin = ax2.twinx()
    
    ax2.bar(x, support, alpha=0.6, color='#9b59b6', label='Sample Count')
    ax2_twin.plot(x, avg_confidence_per_class, 'o-', color='#e67e22', 
                 linewidth=2, markersize=8, label='Avg Confidence')
    
    ax2.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Sample Count', fontsize=12, fontweight='bold', color='#9b59b6')
    ax2_twin.set_ylabel('Avg Confidence', fontsize=12, fontweight='bold', color='#e67e22')
    ax2.set_title(f'{dataset_name} Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.tick_params(axis='y', labelcolor='#9b59b6')
    ax2_twin.tick_params(axis='y', labelcolor='#e67e22')
    ax2_twin.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    for i, (s, c) in enumerate(zip(support, avg_confidence_per_class)):
        ax2.text(i, s + max(support) * 0.02, str(s), ha='center', va='bottom', fontsize=9)
        ax2_twin.text(i, c + 0.03, f'{c:.2f}', ha='center', va='bottom', fontsize=8, color='#e67e22')
    
    plt.tight_layout()
    
    save_path = f'models/class_performance_{dataset_name.lower()}_cnn_lstm.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"클래스 성능 저장: {save_path}")
    
    plt.show()
    plt.close()


def plot_training_history(history):
    """훈련 히스토리 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    # 정확도
    ax1.plot(epochs, history.history['accuracy'], 'b-', 
            label='Training Accuracy', linewidth=2, marker='o', markersize=3)
    ax1.plot(epochs, history.history['val_accuracy'], 'r-', 
            label='Validation Accuracy', linewidth=2, marker='s', markersize=3)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.0])
    
    # 손실
    ax2.plot(epochs, history.history['loss'], 'b-', 
            label='Training Loss', linewidth=2, marker='o', markersize=3)
    ax2.plot(epochs, history.history['val_loss'], 'r-', 
            label='Validation Loss', linewidth=2, marker='s', markersize=3)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/training_history_cnn_lstm.png', dpi=300, bbox_inches='tight')
    logging.info("훈련 히스토리 저장: models/training_history_cnn_lstm.png")
    plt.show()
    plt.close()
