import os
import logging
import warnings

# C++ 로그(InitGoogleLogging 경고) 숨기기
# 0: 모든 로그, 1: INFO, 2: WARNING, 3: ERROR
# 파이썬 코드 실행 전에 환경변수를 설정해야 먹힙니다.
os.environ['GLOG_minloglevel'] = '3'

# 불필요한 경고 메시지 무시 (ccache 경고 제거)
# 이 설정도 paddle 관련 임포트 전에 설정해야 합니다.
warnings.filterwarnings("ignore", category=UserWarning)

from paddleocr import PaddleOCR

def main():
    # 1. 모델 초기화 (한 번만 실행됨)
    # use_textline_orientation=True : 글자가 돌아가 있어도 읽어내는 옵션
    ocr = PaddleOCR(use_textline_orientation=True, lang='korean') 

    # 2. 이미지 경로 (테스트할 이미지)
    img_path = './receipt_test.jpg'

    print(f"--- '{img_path}' 인식 시작 ---")

    # 3. 예측 실행
    result = ocr.predict(img_path)

    # 4. 결과 파싱
    if result and len(result) > 0:
        # v5 모델의 결과 구조 처리
        # 리스트 안에 딕셔너리가 들어있는 구조일 경우
        if isinstance(result, list) and isinstance(result[0], dict):
            data = result[0]
            texts = data.get('rec_texts', [])
            scores = data.get('rec_scores', [])
            
            print(f"\n[총 {len(texts)}개의 텍스트 덩어리를 찾았습니다]\n")

            for i, (text, score) in enumerate(zip(texts, scores)):
                confidence = score * 100
                print(f"[{i+1}] {text} (정확도: {confidence:.2f}%)")
    else:
        print("이미지에서 텍스트를 찾지 못했습니다.")
    
    print("\n--- 인식 종료 ---")


if __name__ == "__main__":
    main()