import os
import joblib
import pandas as pd
from src.data_handler import create_dummy_data_if_needed, load_data_with_comparison, preprocess_data, safe_ttest
from src.ml_pipeline import build_and_evaluate_models
from src.report_generator import generate_report

def main():
    print("="*70)
    print(" 🚀 [MLOps 운영 파이프라인] IT 개발자 AI 도구 활용 영향 분석 시작")
    print("="*70)

    # 1. 환경 설정 및 데이터 준비
    data_file = 'results.csv'
    output_dir = 'project_result'
    os.makedirs(output_dir, exist_ok=True)
    create_dummy_data_if_needed(data_file)

    # 2. 데이터 로딩 비교 (Pandas vs Polars)
    df_raw, p_time, pl_time = load_data_with_comparison(data_file)
    print(f"🕒 Pandas 로딩: {p_time:.5f}초 | Polars 로딩: {pl_time:.5f}초")

    # 3. 전처리
    df_clean = preprocess_data(df_raw)
    print(f"✨ 전처리 완료 (유효 데이터: {len(df_clean)}행)")

    # 4. 통계 분석 (t-test)
    ai_yes_sal = df_clean[df_clean['AI_Status'] == 'Yes']['ConvertedCompYearly']
    ai_no_sal = df_clean[df_clean['AI_Status'] == 'No']['ConvertedCompYearly']
    sal_t, sal_p = safe_ttest(ai_yes_sal, ai_no_sal, "연봉")
    
    sal_interp = "AI 도구 사용자의 연봉이 통계적으로 유의미하게 높습니다." if sal_p < 0.05 and sal_t > 0 else "연봉 차이가 통계적으로 유의미하지 않거나 미사용자가 높습니다."

    ai_yes_sat = df_clean[df_clean['AI_Status'] == 'Yes']['JobSat']
    ai_no_sat = df_clean[df_clean['AI_Status'] == 'No']['JobSat']
    sat_t, sat_p = safe_ttest(ai_yes_sat, ai_no_sat, "만족도")
    
    sat_interp = "AI 도구 사용자의 직무 만족도가 통계적으로 유의미하게 높습니다." if sat_p < 0.05 and sat_t > 0 else "직무 만족도 차이가 유의미하지 않습니다."

    # 5. 다중 머신러닝 모델 학습 (Precision, Recall, F1 포함 평가)
    print("⏳ 다중 머신러닝 모델 학습 및 다각도 지표 평가 중...")
    best_pipe, best_name, best_acc, ml_results_df = build_and_evaluate_models(df_clean)
    
    model_path = os.path.join(output_dir, 'best_developer_pipeline_model.pkl')
    joblib.dump(best_pipe, model_path)
    print(f"💾 최적 모델 저장 완료: {model_path} (모델: {best_name})")

    # 6. Jinja2 기반 정기 문서화 체계 실행
    num_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
    desc_str = df_clean[num_cols[:3]].describe().T[['mean', 'std', '25%', '50%', '75%']].to_string()

    context = {
        'pandas_time': f"{p_time:.5f}",
        'polars_time': f"{pl_time:.5f}",
        'total_rows': len(df_clean),
        'desc_stats': desc_str,
        'sal_yes_mean': f"{ai_yes_sal.mean():,.2f}",
        'sal_no_mean': f"{ai_no_sal.mean():,.2f}",
        'sal_t_stat': f"{sal_t:.4f}",
        'sal_p_val': f"{sal_p:.4e}",
        'sal_interp': sal_interp,
        'sat_yes_mean': f"{ai_yes_sat.mean():.2f}",
        'sat_no_mean': f"{ai_no_sat.mean():.2f}",
        'sat_t_stat': f"{sat_t:.4f}",
        'sat_p_val': f"{sat_p:.4e}",
        'sat_interp': sat_interp,
        'ml_results_table': ml_results_df.to_string(index=False),
        'best_model_name': best_name,
        'best_accuracy': f"{best_acc*100:.2f}",
        'model_filename': model_path
    }

    report_path = os.path.join(output_dir, 'report.md')
    generate_report('templates', 'report_template.md.j2', report_path, context)
    print("="*70)
    print(" 🎉 모든 파이프라인 실행 및 문서화가 성공적으로 완료되었습니다!")
    print("="*70)

if __name__ == '__main__':
    main()