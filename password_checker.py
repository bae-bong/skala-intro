import re


def is_valid_password(password: str) -> bool:
    """비밀번호가 조건(소문자, 대문자, 숫자, 기호 각 1개 이상 포함)을 만족하는지 검사"""
    pattern = re.compile(
        r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()_+\-=\[\]{};:\'",.<>/?\\|`~]).+$'
    )
    return bool(pattern.match(password))


def main():
    while True:
        password = input("비밀번호를 입력하세요 (!quit 입력 시 종료): ")

        if password == "!quit":
            print("프로그램을 종료합니다. 안녕히 가세요!")
            break

        if is_valid_password(password):
            print("사용 가능한 비밀번호입니다.")
        else:
            print("비밀번호는 영문 소문자, 대문자, 숫자, 기호를 각각 최소 1개 이상 포함해야 합니다.")


if __name__ == "__main__":
    main()