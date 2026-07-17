# ------------------------------------------------
# 작성자 : 배은빈
# 작성목적 : Pandas EDA · Polars Lazy · DuckDB SQL 성능 및 문법 비교 실습 practice3
# 작성일 : 2026-07-16
# ------------------------------------------------

import os
import sys
import timeit
import duckdb
import numpy as np
import pandas as pd
import polars as pl

print("==========================================================")
print("  [1단계] Pandas EDA: 기초 탐색 + IQR 이상치 제거 및 예외 처리")
print("==========================================================")

# -------------------------------------------------------------------------
# [기능 설명] 안전한 파일 로드 및 결측치 전처리 함수
# - 목적: CSV 파일 존재 여부를 검증하고, 읽기 오류 시 프로그램이 강제 종료되지 않도록 방어합니다.
# -------------------------------------------------------------------------
file_name = "sales_100k.csv"
clean_file_name = "sales_100k_imputed.csv"

try:
    # 1. 파일 존재 여부 사전 검증 (FileNotFoundError 방어)
    if not os.path.exists(file_name):
        raise FileNotFoundError(
            f"'{file_name}' 파일을 찾을 수 없습니다. 터미널 위치나 파일 경로를 확인해 주세요."
        )

    # 2. 데이터 로드 (빈 파일일 경우 EmptyDataError 방어)
    df = pd.read_csv(file_name)

    if df.empty:
        raise ValueError("불러온 CSV 파일에 데이터가 한 건도 없습니다.")

    print("▶ [성공] 데이터 기본 정보 로드 완료:")
    df.info()

    print("\n▶ 결측치 개수 확인:")
    print(df.isnull().sum())

    # 3. 결측치 전처리 (비즈니스 로직에 따른 예외 데이터 복원)
    # - amount가 누락된 경우 quantity * unit_price로 계산하여 복원
    # - region이 누락된 경우 '미상' 값으로 대체하여 그룹화 시 누락 방지
    df["amount"] = df["amount"].fillna(df["quantity"] * df["unit_price"])
    df["region"] = df["region"].fillna("미상")

except FileNotFoundError as fnf_err:
    print(fnf_err)
    sys.exit(1)  # 파일이 없으면 이후 분석이 불가능하므로 안전하게 프로그램 종료
except pd.errors.EmptyDataError:
    print("CSV 파일이 비어 있거나 손상되었습니다.")
    sys.exit(1)
except Exception as e:
    print(f"데이터 로드 중 알 수 없는 오류 발생: {e}")
    sys.exit(1)


# -------------------------------------------------------------------------
# [기능 설명] IQR(사분위수 범위) 기반 이상치 제거 함수
# - 목적: 통계적으로 극단적인 결제금액(amount)을 필터링하여 분석의 왜곡을 방지합니다.
# -------------------------------------------------------------------------
try:
    Q1 = df["amount"].quantile(0.25)
    Q3 = df["amount"].quantile(0.75)
    IQR = Q3 - Q1

    # 정상 데이터 범위 산출 (Q1 - 1.5*IQR ~ Q3 + 1.5*IQR)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    rows_before = len(df)

    # 정상 범위 내의 데이터만 추출 (.copy()를 통해 SettingWithCopyWarning 예외 방지)
    df_clean = df[
        (df["amount"] >= lower_bound) & (df["amount"] <= upper_bound)
    ].copy()
    rows_after = len(df_clean)

    # [예외 처리] 이상치를 제거했더니 남은 데이터가 없는 경우 방어
    if rows_after == 0:
        raise ValueError("이상치 제거 후 남은 데이터가 0건입니다. IQR 기준을 확인하세요.")

    print(f"\n▶ [IQR 이상치 제거 성공]")
    print(f" - IQR 정상 범위 : {lower_bound:,.1f}원 ~ {upper_bound:,.1f}원")
    print(f" - 제거 전 행 수  : {rows_before:,}행")
    print(f" - 제거 후 행 수  : {rows_after:,}행")
    print(f" - 제거된 이상치  : {rows_before - rows_after:,}행")

    # 다음 단계(Polars, DuckDB)에서 동일한 전처리 데이터를 쓸 수 있도록 임시 파일 저장
    df.to_csv(clean_file_name, index=False)

except ValueError as val_err:
    print(val_err)
    sys.exit(1)
except KeyError:
    print("'amount' 컬럼이 존재하지 않아 이상치를 제거할 수 없습니다.")
    sys.exit(1)


print("\n==========================================================")
print("  [2단계] Pandas Groupby: Named Aggregation")
print("==========================================================")

# -------------------------------------------------------------------------
# [기능 설명] Pandas 다중 그룹 Named Aggregation
# - 목적: 지역 및 카테고리별 매출 통계를 집계하고 컬럼명을 직관적으로 지정합니다.
# -------------------------------------------------------------------------
try:
    pandas_result = (
        df_clean.groupby(["region", "category"])
        .agg(
            total=("amount", "sum"),  # 총매출 합계
            mean=("amount", "mean"),  # 평균 매출
            count=("order_id", "count"),  # 주문 건수
        )
        .reset_index()
        .sort_values(by="total", ascending=False)
    )

    print("▶ Pandas 집계 성공 (Top 5):")
    print(pandas_result.head())

except KeyError as e:
    print(f"그룹화 또는 집계에 필요한 컬럼이 없습니다: {e}")
except Exception as e:
    print(f"Pandas 집계 중 오류 발생: {e}")


print("\n==========================================================")
print("  [3단계] Polars Lazy API 체인 작성 및 실행")
print("==========================================================")

# -------------------------------------------------------------------------
# [기능 설명] Polars Lazy API 파이프라인
# - 목적: 대용량 처리 최적화를 위해 지연 평가(Lazy Evaluation) 방식으로 집계합니다.
# -------------------------------------------------------------------------
try:
    # scan_csv는 실제로 파일을 읽지 않고 실행 계획(Plan)만 수립합니다.
    polars_query = (
        pl.scan_csv(clean_file_name)
        .filter(
            (pl.col("amount") >= lower_bound)
            & (pl.col("amount") <= upper_bound)
        )
        .group_by(["region", "category"])
        .agg(
            [
                pl.col("amount").sum().alias("total"),
                pl.col("amount").mean().alias("mean"),
                pl.col("order_id").count().alias("count"),
            ]
        )
        .sort("total", descending=True)
    )

    # .collect()를 호출하는 순간 예외가 발생할 수 있으므로 try 블록 안에서 실행합니다.
    polars_result = polars_query.collect()

    print("▶ Polars Lazy API 집계 성공 (Top 5):")
    print(polars_result.head(5))

except pl.exceptions.ComputeError as e:
    print(f"Polars 연산 중 오류 발생 (타입 불일치 등): {e}")
except Exception as e:
    print(f"Polars 실행 중 예기치 못한 오류 발생: {e}")


print("\n==========================================================")
print("  [4단계] DuckDB SQL GROUP BY 작성 및 DataFrame 출력")
print("==========================================================")

# -------------------------------------------------------------------------
# [기능 설명] DuckDB 인메모리 SQL 분석
# - 목적: CSV 파일을 DB टेबल처럼 직접 쿼리하여 ANSI SQL 방식으로 집계합니다.
# -------------------------------------------------------------------------
try:
    duckdb_query = f"""
    SELECT 
        region,
        category,
        SUM(amount) AS total,
        AVG(amount) AS mean,
        COUNT(order_id) AS count
    FROM '{clean_file_name}'
    WHERE amount >= {lower_bound} AND amount <= {upper_bound}
    GROUP BY region, category
    ORDER BY total DESC
    """

    duckdb_result = duckdb.query(duckdb_query).to_df()
    print("▶ DuckDB SQL 집계 성공 (Top 5):")
    print(duckdb_result.head())

except duckdb.CatalogException:
    print(f"DuckDB 오류: '{clean_file_name}' 파일을 찾을 수 없거나 테이블 참조가 잘못되었습니다.")
except duckdb.ParserException as e:
    print(f"SQL 문법 오류가 발생했습니다: {e}")
except Exception as e:
    print(f"DuckDB 쿼리 실행 중 오류 발생: {e}")


print("\n==========================================================")
print("  [5단계] timeit 속도 벤치마크 (예외 방어 적용)")
print("==========================================================")

# -------------------------------------------------------------------------
# [기능 설명] 실행 시간 속도 검증 (Benchmarking)
# - 목적: 각 도구별 1회 실행 소요 시간을 공정하게 측정하여 성능을 비교합니다.
# -------------------------------------------------------------------------
NUM_LOOPS = 10
print(f"세 도구 실행 시간 벤치마크 진행 중 (각 {NUM_LOOPS}회 반복)...")


# 측정 대상 함수들 (내부에서 발생하는 오류도 방어)
def run_pandas():
    d = pd.read_csv(clean_file_name)
    d_clean = d[(d["amount"] >= lower_bound) & (d["amount"] <= upper_bound)]
    return (
        d_clean.groupby(["region", "category"])
        .agg(
            total=("amount", "sum"),
            mean=("amount", "mean"),
            count=("order_id", "count"),
        )
        .reset_index()
        .sort_values(by="total", ascending=False)
    )


def run_polars():
    return (
        pl.scan_csv(clean_file_name)
        .filter(
            (pl.col("amount") >= lower_bound)
            & (pl.col("amount") <= upper_bound)
        )
        .group_by(["region", "category"])
        .agg(
            [
                pl.col("amount").sum().alias("total"),
                pl.col("amount").mean().alias("mean"),
                pl.col("order_id").count().alias("count"),
            ]
        )
        .sort("total", descending=True)
        .collect()
    )


def run_duckdb():
    q = f"""
    SELECT region, category, SUM(amount) AS total, AVG(amount) AS mean, COUNT(order_id) AS count
    FROM '{clean_file_name}'
    WHERE amount >= {lower_bound} AND amount <= {upper_bound}
    GROUP BY region, category
    ORDER BY total DESC
    """
    return duckdb.query(q).to_df()


try:
    # NUM_LOOPS가 0 이하일 경우 ZeroDivisionError 방어
    if NUM_LOOPS <= 0:
        raise ValueError("반복 횟수(NUM_LOOPS)는 1 이상이어야 합니다.")

    time_pandas = timeit.timeit(run_pandas, number=NUM_LOOPS) / NUM_LOOPS
    time_polars = timeit.timeit(run_polars, number=NUM_LOOPS) / NUM_LOOPS
    time_duckdb = timeit.timeit(run_duckdb, number=NUM_LOOPS) / NUM_LOOPS

    print("\n--- [최종 평균 실행 시간 비교 결과] ---")
    print(f"1. Pandas      : {time_pandas:.5f} 초")
    print(
        f"2. Polars Lazy : {time_polars:.5f} 초 (Pandas 대비 약 {time_pandas/time_polars:.1f}배 빠름)"
    )
    print(
        f"3. DuckDB SQL  : {time_duckdb:.5f} 초 (Pandas 대비 약 {time_pandas/time_duckdb:.1f}배 빠름)"
    )

except ZeroDivisionError:
    print("나누기 오류: 반복 횟수가 0으로 설정되어 평균 시간을 구할 수 없습니다.")
except Exception as e:
    print(f"벤치마크 측정 중 오류 발생: {e}")
finally:
    # -------------------------------------------------------------------------
    # [기능 설명] cleanup (임시 파일 생성 정리)
    # - 목적: 분석을 위해 임시로 생성했던 디스크의 중간 파일을 깔끔하게 삭제합니다.
    # -------------------------------------------------------------------------
    if os.path.exists(clean_file_name):
        try:
            os.remove(clean_file_name)
            print(f"\n[안내] 임시 생성 파일('{clean_file_name}')을 안전하게 삭제했습니다.")
        except PermissionError:
            print(
                f"\n[경고] '{clean_file_name}' 파일을 다른 프로그램이 사용 중이어서 삭제하지 못했습니다."
            )
        except Exception as e:
            print(f"\n[경고] 임시 파일 삭제 실패: {e}")

print("==========================================================")