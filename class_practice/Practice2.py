# ------------------------------------------------
# 작성자 : 배은빈
# 작성목적 : Python 실습2 - Pydantic v2 + 검증 파이프라인 + 재로딩 확인
# 작성일 : 2026-07-15
# ------------------------------------------------

import csv
import json
import logging
import os
from typing import Optional
from pydantic import BaseModel, Field, ValidationError, field_validator

# ===================================
# 0) logger 초기화 설정
# ===================================
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ===================================
# 1) 예외 처리 + 파일 읽기
# ===================================
def safe_load_csv(file_path: str):  # 파일이 없거나 깨져 있어도 강제 종료되지 않고 안전하게 데이터를 불러오는 함수
    # 완성파일 없으면 None 반환·logger.error, 성공 시 dict 리스트 반환·logger.info, finally에서 '로딩 종료' 출력
    try:
        # 파일 열기
        with open(file_path, mode='r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"파일 '{file_path}'을 성공적으로 불러왔습니다.")   #성공 시 로그 출력
        return data
    except FileNotFoundError:   # 파일 이름이 틀렸거나 파일이 없을 때 실행됨
        logger.error(f"파일 '{file_path}'을 찾을 수 없습니다.")
        return None # 파일 없으면 None 반환
    except json.JSONDecodeError as e:   # 파일은 있지만 괄호가 빠지는 등 JSON 형식이 깨졌을 때 실행됨
        logger.error(f"JSON 파일 형식이 올바르지 않습니다: {e}")
        return None
    except Exception as e:  # 위에서 잡지 못한 알 수 없는 오류 발생 시 실행됨
        logger.error(f'알 수 없는 오류 발생: {e}')
        return None
    finally:    # 무조건 실행
        print('로딩 종료')


# ===================================
# 2) Pydantic v2 스키마 정의
# ===================================
# SalesRecord 모델 완성. date·region: 비어있으면 안 됨. amount: 0 초과. category: 없어도 됨
class SalesRecord(BaseModel):
    # 제공된 데이터의 'month' 키를 'date' 필드로 매핑(alias 사용)
    # min_length=1로 비어있으면 안 되도록 설정(빈 문자열 방지)
    date: str = Field(..., alias='month', min_length=1, description="날짜/월 (비어있으면 안 됨)")
    region: str = Field(..., min_length=1, description="지역 (비어있으면 안 됨)")
    amount: int = Field(..., gt=0, description="판매 금액 (0 초과여야 함)")
    category: Optional[str] = Field(None, description="카테고리 (없어도 됨)")
    

# ===================================
# 3) 검증 파이프라인(valid / errors 분리)
# ===================================
def validate_pipeline(raw_data: list):
    # raw_data를 순회하며 SalesRecord로 변환 성공 시 valid, 실패 시 errors({row, error}) 리스트 반환

    valid = []
    errors = []

    for idx, row in enumerate(raw_data):
        try:
            # Pydantic 모델을 사용하여 데이터 검증
            record = SalesRecord.model_validate(row)
            # 검증 성공 시 딕셔너리로 변환하여 valid 리스트에 추가
            valid.append(record.model_dump(by_alias=True))
        except ValidationError as e:
            # 검증 실패 시 원본 데이터와 에러 내역을 errors 리스트에 추가
            # Checkpoint: ValidationError 발생 시 오류 내용 출력
            logger.warning(
                f"[{idx}번 행 검증 실패] 사유: {e.errors()[0]['msg']}"
            )

            errors.append(
                {
                    "row_index": idx,
                    "row_data": row,
                    "error_details": json.loads(e.json()),  # 에러 상세 내역을 JSON 형식으로 변환하여 저장
                }
            )
        except Exception as e:
            errors.append({'row_index': idx, 'row_data': row, 'error_details': str(e)})
    return valid, errors


# ===================================
# 4) 결과 파일 저장 + 재로딩 확인
# ===================================
def save_and_verify(valid_data: list, errors_data: list):
    # 정상 데이터는 csv로, 에러 데이터는 json으로 저장한 뒤, 파일이 잘 만들어졌는지 다시 확인
    valid_file = 'valid_records.csv'
    error_file = 'error_records.json'

    # 1. Valid 레코드를 CSV 파일로 저장
    if valid_data:
        headers = valid_data[0].keys()
        with open(valid_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(valid_data)
        logger.info(f"Valid 데이터 CSV 저장 완료 : '{valid_file}'")

    # 2. with open(error_file, 'w', encoding='utf-8') as f:
        json.dump(errors_data, f, ensure_ascii=False, indent=4)
    logger.info(f"Error 데이터 JSON 저장 완료 : '{error_file}'")

    # 3. 다시 읽어서 건수 검증(재로딩 확인)
    reloaded_valid_count = 0
    if os.path.exists(valid_file):
        with open(valid_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # 헤더 제외
            reloaded_valid_count = sum(1 for _ in reader)

        with open(error_file, 'r', encoding='utf-8') as f:
            reloaded_errors = json.load(f)
            reloaded_error_count = len(reloaded_errors)
            
        # 검증 결과 출력
        print('--------------------------------')
        print('재로딩 검증 결과')
        print('--------------------------------')
        print(f"Valid 레코드 : {reloaded_valid_count}")
        print(f"Error 레코드 : {reloaded_error_count}")

# ===================================
# 5) 메인 실행부
# ===================================
if __name__ == "__main__":
    # --------------------------------------------------------------
    # safe_load_csv 동작 + assert None 통과 테스트
    # --------------------------------------------------------------
    print("\n>>> [테스트 1] 없는 파일 로딩 테스트 (assert None 통과)")
    missing_data = safe_load_csv("non_existent_file.json")
    # 없는 파일을 넣었으니 missing_data가 None이 나와야 통과
    assert (
        missing_data is None
    ), "Checkpoint 실패: 없는 파일 로드 시 None이어야 합니다!"
    logger.info("-> assert None 통과 완료!\n")

    # --------------------------------------------------------------
    # valid 4건 / errors 3건 assert 통과 테스트
    # --------------------------------------------------------------
    print(">>> [테스트 2] 검증 파이프라인 및 재로딩 테스트")
    # 정상 4건 + 불량 3건 데이터를 인위적으로 만든것
    test_sales_data = [
        # --- 정상 (Valid) 4건 ---
        {"region": "서울", "category": "전자", "amount": 1500, "month": "2024-01"},
        {"region": "부산", "category": "의류", "amount": 800, "month": "2024-01"},
        {"region": "서울", "category": "의류", "amount": 1200, "month": "2024-02"},
        {"region": "대구", "category": "전자", "amount": 950, "month": "2024-01"},
        # --- 에러 (Errors) 3건 ---
        {
            "region": "",
            "category": "전자",
            "amount": 1500,
            "month": "2024-05",
        },  # 에러 1: region 빈 문자열
        {
            "region": "서울",
            "category": "의류",
            "amount": -500,
            "month": "2024-05",
        },  # 에러 2: amount 0 이하
        {
            "region": "   ",
            "category": "식품",
            "amount": 300,
            "month": "2024-05",
        },  # 에러 3: region 공백
    ]

    # 실습용 임시 파일 생성
    test_file_name = "Python_Practice1_Data.json"
    with open(test_file_name, "w", encoding="utf-8") as f:
        json.dump(test_sales_data, f, ensure_ascii=False, indent=4)

    # 1. 파일 읽기
    raw_data = safe_load_csv(test_file_name)

    if raw_data:
        # 2. 검증 파이프라인 실행
        valid_records, error_records = validate_pipeline(raw_data)

        # Checkpoint: valid 4건 / errors 3건 assert 통과
        assert (
            len(valid_records) == 4
        ), f"Valid 건수 불일치: {len(valid_records)}건"
        assert (
            len(error_records) == 3
        ), f"Errors 건수 불일치: {len(error_records)}건"
        logger.info("-> valid 4건 / errors 3건 assert 통과 완료\n")

        # 3. 파일 저장 및 재로딩 확인
        reloaded_valid, reloaded_error = save_and_verify(
            valid_records, error_records
        )

        # Checkpoint: 재로딩 후 len(reloaded)==4 통과
        assert (
            reloaded_valid == 4
        ), "Checkpoint 실패: 재로딩된 Valid 건수가 4건이 아닙니다"
        logger.info("-> 재로딩 후 len(reloaded)==4 assert 통과 완료")
