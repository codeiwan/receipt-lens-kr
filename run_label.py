import os
import sys

# [핵심 해결책] 라이브러리 충돌 에러(OMP Error #15)를 무시하는 설정
# 이 코드가 없으면 윈도우에서 딥러닝 GUI가 자주 튕깁니다.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Python 코드로 PPOCRLabel 강제 실행
from PPOCRLabel.PPOCRLabel import main

if __name__ == "__main__":
    # 언어 설정만 한국어로 주고 실행
    # (내부적으로 sys.argv를 조작해서 인자를 전달하는 효과)
    sys.argv = ["PPOCRLabel", "--lang", "ko"]
    main()