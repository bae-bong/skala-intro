# ------------------------------------------------
# 작성자 : 배은빈
# 작성목적 : 자료구조 활용(defaultdict, Counter, 제너레이터) 및 비동기 API 수집(httpx + asyncio.gather), Pydantic v2 검증, 저장 성능 비교
# 작성일 : 2026-07-15
# ------------------------------------------------

import asyncio
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional
import httpx
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, ConfigDict

# ------------------------------------------------
# 0. 자료구조 체크포인트
# ------------------------------------------------
def run_data_structure_check():
    """자료구조 체크포인트 검증"""
    print("=== [실습 1] 자료구조 체크포인트 검증 ===")
    transactions = [
        {"region": "Seoul", "amount": 5000, "item": "Book"},
        {"region": "Busan", "amount": 3000, "item": "Pen"},
        {"region": "Seoul", "amount": 7000, "item": "Laptop"},
    ]
    
    # defaultdict 활용
    region_total = defaultdict(int)
    for t in transactions:
        region_total[t["region"]] += t["amount"]
    assert region_total["Seoul"] == 12000
    
    # Counter 및 컴프리헨션 활용
    items = [t["item"] for t in transactions]
    most_common = Counter(items).most_common()
    print("✔ 자료구조 체크포인트 통과")

# ------------------------------------------------
# 1. 비동기 수집
# ------------------------------------------------
async def fetch_api(client: httpx.AsyncClient, url: str) -> Any:
    """단일 API 비동기 호출 및 응답 정상 확인"""
    response = await client.get(url)
    response.raise_for_status()  # HTTP 오류 발생 시 예외 발생 (정상 확인)
    return response.json()

async def collect_all_apis() -> Dict[str, Any]:
    """3개 API를 asyncio.gather()로 동시 수집하는 파이프라인"""
    print('\n--- 3개 API 비동기 동시 수집 시작 ---')

    urls = {
        'weather': 'https://api.open-meteo.com/v1/forecast?latitude=37.5665&longitude=126.9780&hourly=temperature_2m,precipitation_probability&forecast_days=3&timezone=Asia/Seoul',
        'country': 'https://countries.dev/alpha/KOR',
        "ip": "http://ip-api.com/json/8.8.8.8"
    }

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            fetch_api(client, urls['weather']),
            fetch_api(client, urls['country']),
            fetch_api(client, urls['ip']),
            return_exceptions=True
        )
    print("✔ 3개 API 비동기 동시 수집 완료")
    return {'weather': results[0], 'country': results[1], 'ip': results[2]}

# ------------------------------------------------
# 2. 스키마 검증
# ------------------------------------------------
class HourlyWeather(BaseModel):
    '''시간별 날씨 스키마'''
    time: List[str]
    temperature_2m: List[float]
    # 강수 확률은 0~100 사이의 값이어야 함
    precipitation_probability: List[int] 
    
class WeatherResponse(BaseModel):
    '''API 최상위 응답 스키마(위경도 범위 검증 포함)'''
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    hourly: HourlyWeather

class CountryResponse(BaseModel):
    name: Any
    region: Optional[str] = "Asia"
    # 실제 API 구조에 맞게 cca2를 Optional로 처리
    cca2: Optional[str] = None 
    model_config = ConfigDict(extra='ignore')

class IpResponse(BaseModel):
    '''IP-API 응답 스키마'''
    query: str
    status: str
    country: str
    regionName: str
    

# ------------------------------------------------
# 3. 스키마 검증 및 데이터 가공
# ------------------------------------------------
def validate_and_process(api_data: Dict[str, Any]) -> pd.DataFrame:
    print('--- Pydantic v2 스키마 검증 ---')
    # 국가 데이터가 리스트라면 첫 번째 요소만 사용
    raw_country = api_data['country']
    if isinstance(raw_country, list): raw_country = raw_country[0]

    weather = WeatherResponse.model_validate(api_data['weather'])
    country = CountryResponse.model_validate(raw_country)
    ip = IpResponse.model_validate(api_data['ip'])
    
    df = pd.DataFrame(weather.hourly.model_dump())
    df['country_name'] = country.name.get('common', 'Korea') if isinstance(country.name, dict) else str(country.name)
    df['ip_region'] = ip.regionName
    return df


def compare_storage_performance(df: pd.DataFrame):
    print("\n--- [3단계] CSV vs Parquet 저장 및 성능 측정 ---")
    
    # 측정 및 저장
    start = time.perf_counter()
    df.to_csv("validated_data.csv", index=False)
    csv_write = time.perf_counter() - start
    
    start = time.perf_counter()
    pd.read_csv("validated_data.csv")
    csv_read = time.perf_counter() - start

    start = time.perf_counter()
    df.to_parquet("validated_data.parquet", index=False, engine="pyarrow")
    parquet_write = time.perf_counter() - start
    
    start = time.perf_counter()
    pd.read_parquet("validated_data.parquet", engine="pyarrow")
    parquet_read = time.perf_counter() - start

    # 결과 출력
    print("[성능 측정 결과]")
    print(f"{'구분':<10} | {'CSV (초)':<12} | {'Parquet (초)':<12}")
    print("-" * 40)
    print(f"{'쓰기(Write)':<10} | {csv_write:<12.5f} | {parquet_write:<12.5f}")
    print(f"{'읽기(Read)':<10} | {csv_read:<12.5f} | {parquet_read:<12.5f}")
    
    speed_diff = csv_read / parquet_read if parquet_read > 0 else 1
    print(f"\n✔ 결과: Parquet가 CSV보다 읽기 속도가 {speed_diff:.2f}배 빠릅니다.")


# =====================================================================
# 메인 실행
# =====================================================================
async def main():
    print("==================================================")
    print("    Day 1 종합 실습 데이터 파이프라인 시작        ")
    print("==================================================")
    
    run_data_structure_check()
    raw_data = await collect_all_apis()
    df = validate_and_process(raw_data)
    compare_storage_performance(df)
    
    print("\n==================================================")
    print("       ✔ 모든 파이프라인 요건 달성 완료!          ")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(main())
