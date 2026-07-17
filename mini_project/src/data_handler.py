import os
import time
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats

# [OS별 한글 폰트 설정]
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin': # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_dummy_data_if_needed(file_path):
    """지정된 데이터 파일이 없을 경우 실습용 가상 샘플 데이터를 자동 생성합니다."""
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        np.random.seed(42)
        n_samples = 1200
        sample_df = pd.DataFrame({
            'AISelect': np.random.choice(['Yes', "No, and I don't plan to", 'No, but I plan to soon'], n_samples, p=[0.55, 0.35, 0.10]),
            'RemoteWork': np.random.choice(['Remote', 'In-person', 'Hybrid (some remote, some in-person)'], n_samples, p=[0.50, 0.20, 0.30]),
            'EdLevel': np.random.choice(['Bachelor’s degree (B.A., B.S., B.Eng., etc.)', 'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)', 'Primary/elementary school'], n_samples),
            'YearsCodePro': np.random.normal(7, 4, n_samples).clip(1, 35)
        })
        base_sal = 55000 + sample_df['YearsCodePro'] * 3500
        ai_bonus = np.where(sample_df['AISelect'] == 'Yes', 16000, 0)
        sample_df['ConvertedCompYearly'] = (base_sal + ai_bonus + np.random.normal(0, 15000, n_samples)).clip(25000, 300000)

        base_sat = np.random.normal(6.2, 1.8, n_samples)
        ai_sat_bonus = np.where(sample_df['AISelect'] == 'Yes', 1.3, 0)
        sample_df['JobSat'] = (base_sat + ai_sat_bonus).round().clip(1, 10)

        sample_df.loc[sample_df.sample(frac=0.08).index, 'ConvertedCompYearly'] = np.nan
        sample_df = pd.concat([sample_df, sample_df.iloc[:25]], ignore_index=True)
        sample_df.to_csv(file_path, index=False)
        print(f"💡 가상 샘플 데이터셋 자동 생성 완료: {file_path}")

def load_data_with_comparison(file_path):
    """Pandas와 Polars로 데이터를 로딩하고 소요 시간을 비교합니다."""
    start_time = time.time()
    df_pandas = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
    pandas_time = time.time() - start_time

    start_time = time.time()
    df_polars = pl.read_excel(file_path) if file_path.endswith('.xlsx') else pl.read_csv(file_path)
    polars_time = time.time() - start_time

    return df_pandas, pandas_time, polars_time

def preprocess_data(df):
    """중복 제거, 파생변수 생성, 이상치 및 결측치를 안전하게 처리합니다."""
    df = df.drop_duplicates().copy()
    df['AI_Status'] = df['AISelect'].apply(lambda x: 'Yes' if x == 'Yes' else 'No')

    if 'ConvertedCompYearly' in df.columns:
        q_low, q_high = df['ConvertedCompYearly'].quantile(0.01), df['ConvertedCompYearly'].quantile(0.99)
        df = df[(df['ConvertedCompYearly'] >= q_low) & (df['ConvertedCompYearly'] <= q_high)]
        df['ConvertedCompYearly'] = df['ConvertedCompYearly'].fillna(df['ConvertedCompYearly'].median())

    for col in ['AISelect', 'RemoteWork', 'EdLevel']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    if 'JobSat' in df.columns:
        df['JobSat'] = df['JobSat'].fillna(df['JobSat'].median())

    return df

def safe_ttest(series_yes, series_no, feature_name="변수"):
    """NaN 및 분산 0으로 인한 에러를 원천 차단하는 안전한 독립표본 t-검정 함수"""
    clean_yes = series_yes.dropna()
    clean_no = series_no.dropna()
    
    if len(clean_yes) < 2 or len(clean_no) < 2:
        print(f"⚠️ [{feature_name}] 표본 수가 부족하여 t-test를 수행할 수 없습니다.")
        return 0.0, 1.0
    if clean_yes.var() == 0 and clean_no.var() == 0:
        print(f"⚠️ [{feature_name}] 두 그룹의 분산이 모두 0이므로 t-test를 수행할 수 없습니다.")
        return 0.0, 1.0
        
    return stats.ttest_ind(clean_yes, clean_no, equal_var=False)

# 🟢 [시각화 기능 추가] 그래프를 화면에 띄우고 파일로 자동 저장하는 함수
def visualize_data(df, output_dir):
    print("\n📈 [시각화 진행] 그래프 창이 뜹니다. 확인 후 창을 닫으면 다음 파이프라인이 진행됩니다.")
    
    # 1. Seaborn 정적 Boxplot
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x='AI_Status', y='ConvertedCompYearly', palette='Set2')
    plt.title('[Seaborn 정적 차트] AI 도구 사용 여부에 따른 연봉 분포 비교', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('AI 도구 사용 여부 (Yes: 현재 사용 중 / No: 미사용 또는 도입 예정)', fontsize=12, fontweight='bold')
    plt.ylabel('달러 환산 연봉 (USD)', fontsize=12, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 이미지 파일로 저장 후 화면에 출력
    save_path = os.path.join(output_dir, 'salary_boxplot.png')
    plt.savefig(save_path, dpi=300)
    print(f"🖼️ Boxplot 이미지 저장 완료: {save_path}")
    plt.show() # 창이 뜨고, 사용자가 닫을 때까지 대기

    # 2. Plotly 인터랙티브 산점도
    if 'YearsCodePro' in df.columns:
        df['YearsCodePro_Num'] = pd.to_numeric(df['YearsCodePro'], errors='coerce')
        df_plotly = df.dropna(subset=['YearsCodePro_Num', 'ConvertedCompYearly']).sort_values(by='YearsCodePro_Num', ascending=True)

        fig = px.scatter(
            df_plotly, x='YearsCodePro_Num', y='ConvertedCompYearly', color='AI_Status',
            title='[Plotly 인터랙티브 차트] 개발 경력(연차)과 연봉의 상관관계',
            labels={
                'YearsCodePro_Num': '전문 개발 경력 (연차 / 오름차순)',
                'ConvertedCompYearly': '달러 환산 연봉 (USD)',
                'AI_Status': 'AI 도구 사용 여부'
            },
            opacity=0.7, template='plotly_white'
        )
        fig.update_layout(
            title_x=0.5,
            title_font_size=16,
            legend_title_text='AI 도구 사용 여부',
            font=dict(family='AppleGothic, Malgun Gothic, sans-serif'),
            xaxis=dict(categoryorder='category ascending')
        )
        # 웹 브라우저 창으로 동적 그래프 출력
        fig.show()