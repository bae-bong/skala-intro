import re

def validate_password(password):
    # 비밀번호 조건 정의
    if len(password) < 8:
        return '비밀번호는 최소 8자 이상이어야 합니다.'
    
    if not re.search(r'[A-Z]', password):
        return '비밀번호는 최소 한 개의 대문자 포함이어야 합니다.'
    
    if not re.search(r'[a-z]', password):
        return '비밀번호는 최소 한 개의 소문자 포함이어야 합니다.'
    
    if not re.search(r'\d', password):
        return '비밀번호는 최소 한 개의 숫자 포함이어야 합니다.'
    
    if not re.search(r'[@$!%*?&]', password):
        return '비밀번호는 최소 한 개의 특수문자(@$!%*?&) 포함이어야 합니다.'

    return '비밀번호가 유효합니다.'

# 사용자 입력 받기
password = input("비밀번호를 입력하세요: ")
result = validate_password(password)
print(result)