# ------------------------------------------------
# 작성자 : 배은빈
# 작성목적 : Python 실습 - 리스트 컴프리헨션, Counter, defaultdict, 제너레이터
# 작성일 : 2026-07-15
# ------------------------------------------------


import json
import sys
from collections import Counter, defaultdict

# json 데이터 불러오기(예외/오류 처리)
try:
    with open("Python_Practice1_Data.json", "r", encoding="utf-8") as f:
        sales = json.load(f)
    print("[시스템] 데이터를 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print("[오류] 'Python_Practice1_Data.json' 파일을 찾을 수 없습니다.")
    sys.exit(1)
except json.JSONDecodeError:
    print("[오류] JSON 파일 형식이 올바르지 않습니다.")
    sys.exit(1)
except Exception as e:
    print(f"[오류] 알 수 없는 오류 발생: {e}")
    sys.exit(1)


# 1) 리스트 컴프리헨션

print('\n---- 1) 리스트 컴프리헨션')
# 1-1 amount ≥ 1000인 거래만 필터링
filtered_sales = [item for item in sales if item['amount'] >= 1000]
print(f"1,000 이상 거래 : {filtered_sales}")

# 1-2 지역별 총매출 dict를 컴프리헨션으로
regions = {item['region'] for item in sales}
region_totals = {
    region: sum(item['amount'] for item in sales if item['region'] == region)
    for region in regions
}
print(f"지역별 총매출 : {region_totals}")


# 2) Counter + defaultdict
# - Counter로 지역별 거래 건수를, defaultdict로 카테고리별 amount 리스트

print('\n---- 2) Counter + defaultdict')
# Counter로 지역별 거래 건수 집계
region_counts = Counter(item['region'] for item in sales)

# defaultdict로 카테고리별 amount 리스트 모으기
category_amounts = defaultdict(list)
for item in sales:
    category_amounts[item['category']].append(item['amount'])

print(f"카테고리별 amount 리스트 : {category_amounts}")


# 3) 제너레이터 — 메모리 비교
# amount > 1000 인 행만 yield 하는 제너레이터를 작성하고, 리스트 버전과 메모리 크기를 비교

print('\n---- 3) 제너레이터 — 메모리 비교')
# amout > 1000 인 행만 yield 하는 제너레이터
gen_sales = (item for item in sales if item['amount'] > 1000)

# 비교를 위한 리스트 컴프리헨션
list_sales = [item for item in sales if item['amount'] > 1000]

# 메모리 크기 비교
print(f"제너레이터 메모리 크기: {sys.getsizeof(gen_sales)} bytes")
print(f"리스트 메모리 크기: {sys.getsizeof(list_sales)} bytes")


# 4) 종합 - 월별 카테고리 매출 집계
# sales 데이터를 month·category 기준으로 그룹핑해 총매출 dict를 완성 (컴프리헨션 + defaultdict)

print('\n---- 4) 종합 - 월별 카테고리 매출 집계')
# month와 category 기준으로 그룹핑해서 총매출 dict 생성
monthly_category_sales = defaultdict(int)

for item in sales:
    key = (item['month'], item['category'])
    monthly_category_sales[key] += item['amount']

# 정렬
sorted_result = {k: monthly_category_sales[k] for k in sorted(monthly_category_sales)}

for (month, category), total_amount in sorted_result.items():
    print(f"총매출: {total_amount:,}원")