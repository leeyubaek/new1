"""
상체 데이터 CSV 생성 및 분석 도구
상체 랜드마크와 각도 데이터만 추출하여 정리된 CSV 파일을 생성합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def create_upper_body_csv(input_csv='data/pose_data.csv', output_csv='data/upper_body_data.csv'):
    """상체 데이터만 추출하여 새로운 CSV 생성"""
    
    # 원본 데이터 로드
    df = pd.read_csv(input_csv)
    logging.info(f"원본 데이터 로드 완료: {len(df)}개 샘플")
    
    # 상체 관련 컬럼만 선택
    upper_body_columns = [
        'image_name', 'label',
        # 각도 데이터
        'neck_angle', 'shoulder_angle', 'hip_angle', 'torso_angle',
        # 상체 랜드마크 좌표 (정규화)
        'nose_x', 'nose_y',
        'left_shoulder_x', 'left_shoulder_y',
        'right_shoulder_x', 'right_shoulder_y', 
        'left_hip_x', 'left_hip_y',
        'right_hip_x', 'right_hip_y',
        # visibility 정보
        'nose_visibility',
        'left_shoulder_visibility', 'right_shoulder_visibility',
        'left_hip_visibility', 'right_hip_visibility'
    ]
    
    # 존재하는 컬럼만 선택
    available_columns = [col for col in upper_body_columns if col in df.columns]
    upper_body_df = df[available_columns].copy()
    
    # 각도 데이터 반올림 (소수점 2자리)
    angle_columns = ['neck_angle', 'shoulder_angle', 'hip_angle', 'torso_angle']
    for col in angle_columns:
        if col in upper_body_df.columns:
            upper_body_df[col] = upper_body_df[col].round(2)
    
    # 좌표 데이터 반올림 (소수점 4자리)
    coord_columns = [col for col in upper_body_df.columns if col.endswith('_x') or col.endswith('_y')]
    for col in coord_columns:
        upper_body_df[col] = upper_body_df[col].round(4)
    
    # visibility 반올림 (소수점 3자리)
    visibility_columns = [col for col in upper_body_df.columns if col.endswith('_visibility')]
    for col in visibility_columns:
        upper_body_df[col] = upper_body_df[col].round(3)
    
    # CSV 저장
    upper_body_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    logging.info(f"상체 데이터 CSV 저장 완료: {output_csv}")
    
    return upper_body_df

def analyze_upper_body_data(df):
    """상체 데이터 통계 분석"""
    
    analysis_results = {}
    
    # 기본 정보
    analysis_results['총 샘플 수'] = len(df)
    analysis_results['라벨 분포'] = df['label'].value_counts().to_dict()
    
    # 각도 통계
    angle_columns = ['neck_angle', 'shoulder_angle', 'hip_angle', 'torso_angle']
    angle_stats = {}
    
    for col in angle_columns:
        if col in df.columns:
            angle_stats[col] = {
                '평균': df[col].mean().round(2),
                '표준편차': df[col].std().round(2),
                '최솟값': df[col].min().round(2),
                '최댓값': df[col].max().round(2),
                '중앙값': df[col].median().round(2)
            }
    
    analysis_results['각도 통계'] = angle_stats
    
    # 라벨별 각도 통계
    label_stats = {}
    for label in df['label'].unique():
        if label == 'unlabeled':
            continue
            
        label_data = df[df['label'] == label]
        label_angles = {}
        
        for col in angle_columns:
            if col in df.columns:
                label_angles[col] = {
                    '평균': label_data[col].mean().round(2),
                    '표준편차': label_data[col].std().round(2)
                }
        
        label_stats[label] = label_angles
    
    analysis_results['라벨별 각도 통계'] = label_stats
    
    return analysis_results

def print_summary_table(df, analysis_results):
    """요약 테이블 출력"""
    
    print("=" * 80)
    print("🤖 상체 자세 데이터 분석 결과")
    print("=" * 80)
    
    # 기본 정보
    print(f"\n📊 기본 정보:")
    print(f"   • 총 샘플 수: {analysis_results['총 샘플 수']:,}개")
    print(f"   • 컬럼 수: {len(df.columns)}개")
    
    # 라벨 분포
    print(f"\n🏷️  라벨 분포:")
    for label, count in analysis_results['라벨 분포'].items():
        percentage = (count / analysis_results['총 샘플 수']) * 100
        print(f"   • {label}: {count:,}개 ({percentage:.1f}%)")
    
    # 각도 통계 테이블
    print(f"\n📐 각도 통계 요약:")
    print("-" * 80)
    print(f"{'각도 유형':<12} {'평균':<8} {'표준편차':<8} {'최솟값':<8} {'최댓값':<8} {'중앙값':<8}")
    print("-" * 80)
    
    angle_names = {
        'neck_angle': '목 각도',
        'shoulder_angle': '어깨 각도', 
        'hip_angle': '허리 각도',
        'torso_angle': '상체 각도'
    }
    
    for col, stats in analysis_results['각도 통계'].items():
        name = angle_names.get(col, col)
        print(f"{name:<12} {stats['평균']:<8.1f} {stats['표준편차']:<8.1f} "
              f"{stats['최솟값']:<8.1f} {stats['최댓값']:<8.1f} {stats['중앙값']:<8.1f}")
    
    # 라벨별 비교
    print(f"\n🔍 라벨별 각도 비교:")
    print("-" * 60)
    
    for label, label_stats in analysis_results['라벨별 각도 통계'].items():
        print(f"\n📋 {label.upper()} 자세:")
        for col, stats in label_stats.items():
            name = angle_names.get(col, col)
            print(f"   • {name}: 평균 {stats['평균']:.1f}° (±{stats['표준편차']:.1f}°)")

def create_sample_preview(df, n_samples=10):
    """샘플 데이터 미리보기"""
    
    print(f"\n👀 샘플 데이터 미리보기 (상위 {n_samples}개):")
    print("=" * 120)
    
    # 주요 컬럼만 선택해서 보여주기
    preview_columns = ['image_name', 'label', 'neck_angle', 'shoulder_angle', 'hip_angle', 'torso_angle']
    available_preview = [col for col in preview_columns if col in df.columns]
    
    sample_df = df[available_preview].head(n_samples)
    
    # 컬럼명 한글로 변경
    column_mapping = {
        'image_name': '이미지명',
        'label': '라벨',
        'neck_angle': '목각도',
        'shoulder_angle': '어깨각도',
        'hip_angle': '허리각도',
        'torso_angle': '상체각도'
    }
    
    display_df = sample_df.rename(columns=column_mapping)
    
    # 테이블 형태로 출력
    print(display_df.to_string(index=False, float_format='%.1f'))
    print("=" * 120)

def main():
    """메인 실행 함수"""
    
    try:
        # 1. 상체 데이터 CSV 생성
        logging.info("상체 데이터 CSV 생성 시작...")
        upper_body_df = create_upper_body_csv()
        
        # 2. 데이터 분석
        logging.info("데이터 분석 중...")
        analysis_results = analyze_upper_body_data(upper_body_df)
        
        # 3. 결과 출력
        print_summary_table(upper_body_df, analysis_results)
        
        # 4. 샘플 데이터 미리보기
        create_sample_preview(upper_body_df)
        
        # 5. 통계 CSV 저장
        stats_data = []
        for label, label_stats in analysis_results['라벨별 각도 통계'].items():
            for angle, stats in label_stats.items():
                stats_data.append({
                    'label': label,
                    'angle_type': angle,
                    'mean': stats['평균'],
                    'std': stats['표준편차']
                })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_csv('data/upper_body_statistics.csv', index=False, encoding='utf-8-sig')
            logging.info("상세 통계 저장 완료: data/upper_body_statistics.csv")
        
        print(f"\n✅ 상체 데이터 처리 완료!")
        print(f"📁 생성된 파일들:")
        print(f"   • data/upper_body_data.csv - 정리된 상체 데이터")
        print(f"   • data/upper_body_statistics.csv - 통계 데이터")
        
    except Exception as e:
        logging.error(f"처리 중 오류 발생: {e}")
        raise

if __name__ == '__main__':
    main()