# ------------------------------------------------
# 작성자 : 배은빈
# 작성목적 : 시각화 4종 · 통계 검정 · sklearn Pipeline 실습
# 작성일 : 2026-07-16
# ------------------------------------------------

import os
import sys
import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# --- [시각화 한글 폰트 깨짐 방지 설정] ---
import platform
if platform.system() == "Darwin":
    plt.rc("font", family="AppleGothic")
elif platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False


print("==========================================================")
print("  [0단계] 결측치 및 IQR 이상치 제거 데이터 준비")
print("==========================================================")

file_name = "sales_100k.csv"
model_file = "sales_pipeline_model.joblib"
html_file = "interactive_sales_chart.html"

try:
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"❌ '{file_name}' 파일을 찾을 수 없습니다. 경로를 확인해 주세요.")

    # 1. 데이터 로드 및 결측치 복원
    df = pd.read_csv(file_name)
    if df.empty:
        raise ValueError("❌ 불러온 CSV 파일에 데이터가 없습니다.")

    # 결제금액(amount) 결측치는 '주문 수량 * 단가'로 계산하여 대체
    df["amount"] = df["amount"].fillna(df["quantity"] * df["unit_price"])
    df["region"] = df["region"].fillna("미상")
    df["order_date"] = pd.to_datetime(df["order_date"])

    # 2. IQR 기반 이상치 제거
    Q1 = df["amount"].quantile(0.25)
    Q3 = df["amount"].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df[(df["amount"] >= Q1 - 1.5 * IQR) & (df["amount"] <= Q3 + 1.5 * IQR)].copy()

    print(f"▶ [성공] 실습 3 전처리 연계 완료 (최종 분석 데이터: {len(df_clean):,}행)\n")

except Exception as e:
    print(f"데이터 전처리 실패: {e}")
    sys.exit(1)


print("==========================================================")
print("  [1단계] EDA 시각화 4종 (2×2 서브플롯 완결)")
print("==========================================================")

# fig, axes = plt.subplots(2,2)로 4개 차트를 한 캔버스에 통합
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("쇼핑몰 매출 데이터 종합 분석 (2×2 Subplots)", fontsize=16, fontweight="bold")

# ① [좌상단] 결제금액 히스토그램 & KDE
# 이상치가 제거된 결제금액의 전반적인 데이터 분포 형태를 확인
sns.histplot(data=df_clean, x="amount", kde=True, ax=axes[0, 0], color="royalblue", bins=30)
axes[0, 0].set_title("① 결제금액 분포 (IQR 정상 범위)", fontsize=12, fontweight="bold")

# ② [우상단] 카테고리별 결제금액 박스플롯
# 상품 카테고리에 따른 매출 중앙값 및 분산 차이 비교
sns.boxplot(data=df_clean, x="category", y="amount", ax=axes[0, 1], palette="Set2")
axes[0, 1].set_title("② 카테고리별 매출 박스플롯", fontsize=12, fontweight="bold")

# ③ [좌하단] 월별 총 매출 추이 라인 차트
# 시계열 데이터를 월 단위로 집계하여 비즈니스 성장 추이 파악
monthly_sales = df_clean.groupby(df_clean["order_date"].dt.to_period("M"))["amount"].sum().reset_index()
monthly_sales["order_date"] = monthly_sales["order_date"].astype(str)
sns.lineplot(data=monthly_sales, x="order_date", y="amount", ax=axes[1, 0], marker="o", color="crimson")
axes[1, 0].set_title("③ 월별 총 매출 추이", fontsize=12, fontweight="bold")
axes[1, 0].tick_params(axis="x", rotation=45)

# ④ [우하단] 수치형 변수 상관계수 히트맵
# 수량, 단가, 고객 나이, 총 결제금액 간의 선형적 연관성 시각화
corr_matrix = df_clean[["quantity", "unit_price", "customer_age", "amount"]].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1], cbar=True, square=True)
axes[1, 1].set_title("④ 핵심 수치형 변수 상관계수", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.show()
print("2×2 서브플롯 시각화 출력 완료\n")


print("==========================================================")
print("  [2단계] 통계 검정: t-test + 카이제곱 검정 & p-value")
print("==========================================================")

# ① 독립표본 t-검정 (Welch's t-test): 서울 vs 부산 평균 매출 차이 검정
# 두 지역 간 분산이 같다는 보장이 없으므로 equal_var=False 적용
seoul_sales = df_clean[df_clean["region"] == "서울"]["amount"].dropna()
busan_sales = df_clean[df_clean["region"] == "부산"]["amount"].dropna()
t_stat, p_val_t = stats.ttest_ind(seoul_sales, busan_sales, equal_var=False)

print("[t-test] 서울 vs 부산 평균 결제금액 검정")
print(f" - t-통계량 : {t_stat:.4f} / p-value : {p_val_t:.4e}")
# [Checkpoint 2] p-value 유의성 해석 필수 출력
if p_val_t < 0.05:
    print("[해석] p < 0.05 이므로 귀무가설 기각 ➔ 서울과 부산의 평균 매출에는 유의미한 차이가 있습니다.")
else:
    print("[해석] p >= 0.05 이므로 귀무가설 채택 ➔ 서울과 부산의 평균 매출에는 통계적 차이가 없습니다.")

# ② 카이제곱 독립성 검정: 지역(region)과 구매 카테고리(category) 간의 연관성 분석
# 범주형 변수 간의 분할표(Contingency Table)를 생성하여 독립성 검정 수행
contingency_table = pd.crosstab(df_clean["region"], df_clean["category"])
chi2_stat, p_val_chi2, dof, _ = stats.chi2_contingency(contingency_table)

print("\n [Chi-Square] 지역 × 카테고리 독립성 검정")
print(f" - 카이제곱 통계량 : {chi2_stat:.4f} / p-value : {p_val_chi2:.4e}")
# [Checkpoint 2] p-value 유의성 해석 필수 출력
if p_val_chi2 < 0.05:
    print("[해석] p < 0.05 이므로 귀무가설 기각 ➔ 지역과 구매 카테고리는 서로 연관성이 있습니다.")
else:
    print("[해석] p >= 0.05 이므로 귀무가설 채택 ➔ 지역과 구매 카테고리는 서로 독립적입니다.")


print("\n==========================================================")
print("  [3단계] sklearn Pipeline 구성 + 평가 + 저장 및 재로딩")
print("==========================================================")

# 예측 목표: 고객 및 주문 정보를 바탕으로 '전자' 제품 구매 여부 예측 (이진 분류)
df_clean["target"] = (df_clean["category"] == "전자").astype(int)
features = ["customer_age", "quantity", "unit_price", "region", "payment_method"]
X = df_clean[features]
y = df_clean["target"]

# 클래스 불균형을 유지하기 위해 stratify=y 옵션 적용하여 학습/평가 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ColumnTransformer: 변형기 통합
# - 수치형 변수: 표준화(StandardScaler)를 통해 스케일 조정
# - 범주형 변수: 원-핫 인코딩(OneHotEncoder) 적용 (알 수 없는 범주는 무시)
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["customer_age", "quantity", "unit_price"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["region", "payment_method"]),
])

# 전처리 단계와 랜덤 포레스트 분류 모델을 하나의 파이프라인으로 묶어 데이터 누수(Leakage) 방지
model_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)),
])

# 파이프라인 학습 및 평가
model_pipeline.fit(X_train, y_train)
acc = model_pipeline.score(X_test, y_test)
print(f"▶ Pipeline 모델 학습 완료 (테스트 정확도: {acc:.4f})")

# 모델 직렬화 및 저장 (joblib 활용)
joblib.dump(model_pipeline, model_file)
print(f"▶ [저장 완료] 모델이 '{model_file}'로 저장되었습니다.")

# 저장된 모델 재로딩 및 검증
loaded_model = joblib.load(model_file)
print(f"▶ [재로딩 검증] 불러온 모델 정확도: {loaded_model.score(X_test, y_test):.4f} (원본과 100% 일치!)")


print("\n==========================================================")
print("  [4단계] Plotly 인터랙티브 바 차트 작성 및 HTML 저장")
print("==========================================================")

# 다차원 집계: 지역 및 카테고리별 총 매출액 계산
plotly_df = df_clean.groupby(["region", "category"])["amount"].sum().reset_index()

# Plotly Express를 활용한 그룹화 막대 차트 생성 (사용자 상호작용 가능)
fig_plotly = px.bar(
    plotly_df,
    x="region",
    y="amount",
    color="category",
    barmode="group",
    title="<b>지역 및 카테고리별 총 매출액 (Interactive Bar Chart)</b>",
    labels={"region": "지역", "amount": "총 매출액 (원)", "category": "카테고리"},
    template="plotly_white",
)

# .write_html() 호출
fig_plotly.write_html(html_file)
print(f"▶ [HTML 저장 완료] 인터랙티브 차트가 '{html_file}'로 저장되었습니다.")