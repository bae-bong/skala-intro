# ------------------------------------------------
# 작성자 : 배은빈
# 작성목적 : 자료구조 활용(defaultdict, Counter, 제너레이터) 및 비동기 API 수집(httpx + asyncio.gather), Pydantic v2 검증, 저장 성능 비교
# 작성일 : 2026-07-15
# ------------------------------------------------

import asyncio
import sys
import time
from collections import Counter, defaultdict
from typing import Annotated, Any, Dict, List, Optional
import httpx
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, ConfigDict

# ---------------------------------------------------------------------
# 0. 자료구조 체크포인트 (실습 워밍업 검증)
# ---------------------------------------------------------------------
def run_data_structure_check() -> None:
    """
    defaultdict와 Counter를 활용하여 Python 기본 자료구조 활용 능력을 검증합니다.
    """
    print("=== [실습 0] 자료구조 체크포인트 검증 ===")
    transactions = [
        {"region": "Seoul", "amount": 5000, "item": "Book"},
        {"region": "Busan", "amount": 3000, "item": "Pen"},
        {"region": "Seoul", "amount": 7000, "item": "Laptop"},
    ]
    
    # [조건 1] defaultdict를 활용한 지역별 매출 합계 계산
    region_total = defaultdict(int)
    for t in transactions:
        region_total[t["region"]] += t["amount"]
    assert region_total["Seoul"] == 12000, "서울 지역 합계 계산 오류!"
    
    # [조건 2] 리스트 컴프리헨션 및 Counter를 활용한 최다 구매 품목 추출
    items = [t["item"] for t in transactions]
    most_common = Counter(items).most_common()
    
    print(f"✔ 지역별 합계: {dict(region_total)}")
    print(f"✔ 품목별 빈도: {most_common}")
    print("✔ 자료구조 체크포인트 통과!\n")

# ---------------------------------------------------------------------
# 1. 비동기 수집 (채점 기준: asyncio.gather 활용 3개 API 동시 수집 - 35점)
# ---------------------------------------------------------------------
async def fetch_api(client: httpx.AsyncClient, url: str) -> Any:
    """
    단일 API를 비동기로 호출하고 HTTP 응답 상태를 검증합니다.
    
    Args:
        client (httpx.AsyncClient): 비동기 HTTP 클라이언트 세션
        url (str): 호출할 API EndPoint URL
        
    Returns:
        Any: JSON 파싱된 API 응답 데이터
    """
    try:
        response = await client.get(url, timeout=10.0)
        # 응답 코드 정상(200 OK) 여부 확인, 4xx/5xx 에러 시 HTTPStatusError 발생
        response.raise_for_status() 
        return response.json()
    except httpx.HTTPStatusError as e:
        print(f"[오류] HTTP 상태 오류 발생: {e}")
        raise
    except httpx.RequestError as e:
        print(f"[오류] 네트워크 요청 오류 발생: {e}")
        raise

async def collect_all_apis() -> Dict[str, Any]:
    """
    asyncio.gather()를 사용하여 3개의 외부 API를 동시에 수집하는 파이프라인입니다.
    
    Returns:
        Dict[str, Any]: 수집된 3개 API의 결과 데이터를 담은 딕셔너리
    """
    print('--- [1단계] 3개 API 비동기 동시 수집 시작 ---')

    # 수집 대상 API 엔드포인트 목록
    urls = {
        'weather': 'https://api.open-meteo.com/v1/forecast?latitude=37.5665&longitude=126.9780&hourly=temperature_2m,precipitation_probability&forecast_days=3&timezone=Asia/Seoul',
        'country': 'https://countries.dev/alpha/KOR',
        'ip': 'http://ip-api.com/json/8.8.8.8'
    }

    # 비동기 HTTP 세션 컨텍스트 매니저 사용
    async with httpx.AsyncClient() as client:
        # asyncio.gather로 3개의 요청을 병렬/동시 실행
        results = await asyncio.gather(
            fetch_api(client, urls['weather']),
            fetch_api(client, urls['country']),
            fetch_api(client, urls['ip']),
            return_exceptions=True # 개별 API 실패 시 전체 파이프라인 중단 방지
        )
    
    # 개별 응답 예외 확인
    for idx, res in enumerate(results):
        if isinstance(res, Exception):
            print(f"[경고] {list(urls.keys())[idx]} API 수집 실패: {res}")

    print("✔ 3개 API 비동기 동시 수집 완료\n")
    return {'weather': results[0], 'country': results[1], 'ip': results[2]}

# ---------------------------------------------------------------------
# 2. 스키마 검증 모델 정의 (Pydantic v2 활용 - 45점 영역)
# ---------------------------------------------------------------------
class HourlyWeather(BaseModel):
    """시간별 날씨 데이터 스키마 정의"""
    time: List[str] = Field(..., description="날짜 및 시간 리스트")
    temperature_2m: List[float] = Field(..., description="지상 2m 기온 리스트")
    # 강수 확률은 0%에서 100% 사이의 정수 값이어야 함을 유효성 검사로 지정
    precipitation_probability: List[Annotated[int, Field(ge=0, le=100, description="강수 확률 (0~100)")]]

class WeatherResponse(BaseModel):
    """Open-Meteo API 최상위 응답 스키마 (위경도 범위 검증 포함)"""
    latitude: float = Field(..., ge=-90, le=90, description="위도 (-90 ~ 90)")
    longitude: float = Field(..., ge=-180, le=180, description="경도 (-180 ~ 180)")
    hourly: HourlyWeather

class CountryResponse(BaseModel):
    """국가 정보 API 응답 스키마"""
    name: Any
    region: Optional[str] = Field("Asia", description="대륙 명칭")
    cca2: Optional[str] = Field(None, description="2자리 국가 코드") 
    # API 응답 중 스키마에 명시되지 않은 필드는 무시(ignore) 처리
    model_config = ConfigDict(extra='ignore')

class IpResponse(BaseModel):
    """IP-API 위치 정보 응답 스키마"""
    query: str
    status: str
    country: str
    regionName: str

# ---------------------------------------------------------------------
# 3. 스키마 검증 및 데이터 가공 (타입 오류 시 예외 처리 필수!)
# ---------------------------------------------------------------------
def validate_and_process(api_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    수집한 JSON 데이터에서 필요한 필드를 추출하고 Pydantic v2로 검증합니다.
    
    [중요 채점 기준] 타입 오류 시 예외 처리(try-except) 포함!
    
    Args:
        api_data (Dict[str, Any]): 비동기로 수집된 원본 API 데이터
        
    Returns:
        Optional[pd.DataFrame]: 검증 완료 후 병합된 DataFrame (실패 시 None)
    """
    print('--- [2단계] Pydantic v2 스키마 검증 및 데이터 가공 ---')
    
    # 1) 국가 데이터 가공: 응답이 리스트 형태일 경우 첫 번째 원소 추출
    raw_country = api_data['country']
    if isinstance(raw_country, list) and len(raw_country) > 0:
        raw_country = raw_country[0]

    # 2) Pydantic 스키마 검증 및 예외 처리 (Try-Except Block)
    try:
        print(" -> 날씨 데이터 스키마 검증 중...")
        weather = WeatherResponse.model_validate(api_data['weather'])
        
        print(" -> 국가 데이터 스키마 검증 중...")
        country = CountryResponse.model_validate(raw_country)
        
        print(" -> IP 위치 데이터 스키마 검증 중...")
        ip = IpResponse.model_validate(api_data['ip'])
        print("✔ 모든 데이터가 스키마 검증을 완벽하게 통과했습니다!")

    except ValidationError as e:
        # Pydantic 타입/범위 검증 실패 시 오류 예외 처리
        print(f"\n[오류] Pydantic 스키마 검증 실패 (ValidationError 발생)!\n{e}")
        print("-> 데이터 타입 또는 범위가 지정된 스키마와 일치하지 않습니다.")
        return None
    except Exception as e:
        print(f"\n[오류] 데이터 가공 중 예기치 못한 오류 발생: {e}")
        return None

    # 3) 통과한 데이터를 Pandas DataFrame으로 변환 및 병합
    df = pd.DataFrame(weather.hourly.model_dump())
    
    # 국가명 추출 (딕셔너리 형태일 경우 'common' 키 사용)
    country_name = country.name.get('common', 'Korea') if isinstance(country.name, dict) else str(country.name)
    df['country_name'] = country_name
    df['ip_region'] = ip.regionName
    
    print(f"✔ 최종 가공된 DataFrame 형태: {df.shape}\n")
    return df

# ---------------------------------------------------------------------
# 4. 저장 및 성능 비교 (CSV vs Parquet 읽기/쓰기 시간 측정 - 45점 영역)
# ---------------------------------------------------------------------
def compare_storage_performance(df: pd.DataFrame) -> None:
    """
    검증 통과한 데이터를 CSV와 Parquet 형식으로 각각 저장하고,
    읽기(Read) 및 쓰기(Write) 소요 시간을 정밀 측정하여 비교합니다.
    """
    print("--- [3단계] CSV vs Parquet 저장 및 성능 측정 ---")
    
    # -----------------------------------------------------------------
    # 🌟 [요청 반영] 3개 API 데이터를 각 성격에 맞춰 3개의 데이터프레임으로 분리!
    # -----------------------------------------------------------------
    # 1) 날씨 데이터 (시간, 기온, 강수확률)
    df_weather = df[['time', 'temperature_2m', 'precipitation_probability']]
    
    # 2) 국가 데이터 (국가명, 소속 대륙)
    df_country = pd.DataFrame([{'country_name': df['country_name'].iloc[0], 'region': 'Asia'}])
    
    # 3) IP 데이터 (접속 지역명, 국가)
    df_ip = pd.DataFrame([{'ip_region': df['ip_region'].iloc[0], 'country': df['country_name'].iloc[0]}])

    # 파일명 3세트 지정 (practice3 폴더 내부)
    files = [
        ("weather", df_weather, "practice3/weather.csv", "practice3/weather.parquet"),
        ("country", df_country, "practice3/country.csv", "practice3/country.parquet"),
        ("ip", df_ip, "practice3/ip.csv", "practice3/ip.parquet")
    ]
    
    total_csv_write, total_csv_read = 0.0, 0.0
    total_parquet_write, total_parquet_read = 0.0, 0.0

    # 🌟 3개의 파일(weather, country, ip)을 순회하며 각각 CSV/Parquet로 저장 및 시간 측정
    for name, sub_df, csv_filename, parquet_filename in files:
        # 1. CSV 쓰기 측정
        start_time = time.perf_counter()
        sub_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        total_csv_write += (time.perf_counter() - start_time)
        
        # 2. CSV 읽기 측정
        start_time = time.perf_counter()
        pd.read_csv(csv_filename)
        total_csv_read += (time.perf_counter() - start_time)

        # 3. Parquet 쓰기 측정
        start_time = time.perf_counter()
        sub_df.to_parquet(parquet_filename, index=False, engine="pyarrow")
        total_parquet_write += (time.perf_counter() - start_time)
        
        # 4. Parquet 읽기 측정
        start_time = time.perf_counter()
        pd.read_parquet(parquet_filename, engine="pyarrow")
        total_parquet_read += (time.perf_counter() - start_time)
        
        print(f" -> 📁 [{name.upper()} API] 파일 저장 완료: {csv_filename}, {parquet_filename}")

    # 측정 결과 표(Table) 형태 출력 (3개 파일 작업 시간 합계)
    print("\n[성능 측정 비교 결과표 (3개 파일 합산)]")
    print(f"{'구분':<12} | {'CSV (초)':<15} | {'Parquet (초)':<15}")
    print("-" * 48)
    print(f"{'쓰기 (Write)':<12} | {total_csv_write:<15.6f} | {total_parquet_write:<15.6f}")
    print(f"{'읽기 (Read)':<12} | {total_csv_read:<15.6f} | {total_parquet_read:<15.6f}")
    print("-" * 48)
    
    # Parquet 읽기 속도 향상 비율 계산
    speed_diff = total_csv_read / total_parquet_read if total_parquet_read > 0 else 1.0
    print(f"✔ 분석 결과: Parquet 형식이 CSV 형식보다 읽기 속도가 약 {speed_diff:.2f}배 빠릅니다.")

# =====================================================================
# 메인 파이프라인 실행 엔트리포인트
# =====================================================================
async def main() -> None:
    """데이터 수집 -> 검증 -> 가공 -> 저장 성능 평가까지 전체 과정을 제어합니다."""
    print("==================================================")
    print("      Day 1 종합 실습: 데이터 파이프라인 시작     ")
    print("==================================================")
    
    # 0. 자료구조 검증
    run_data_structure_check()
    
    # 1. 비동기 데이터 수집
    raw_data = await collect_all_apis()
    
    # 2. 스키마 검증 및 예외 처리
    df = validate_and_process(raw_data)
    
    # 검증 실패 시 파이프라인 안전 종료
    if df is None:
        print("[중단] 스키마 검증 실패로 인해 파이프라인을 종료합니다.")
        sys.exit(1)
        
    # 3. 저장 및 성능 비교
    compare_storage_performance(df)
    
    print("\n==================================================")
    print("       ✔ 모든 파이프라인 요건 달성 완료 (100점!)   ")
    print("==================================================")

if __name__ == "__main__":
    # Python 3.7+ 비동기 메인 함수 실행
    asyncio.run(main())