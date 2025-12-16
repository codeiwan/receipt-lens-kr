import os
import warnings

import cv2
import numpy as np

# C++ 로그(InitGoogleLogging 경고) 숨기기
# 0: 모든 로그, 1: INFO, 2: WARNING, 3: ERROR
# 파이썬 코드 실행 전에 환경변수를 설정해야 먹힙니다.
os.environ['GLOG_minloglevel'] = '3'

# 불필요한 경고 메시지 무시 (ccache 경고 제거)
# 이 설정도 paddle 관련 임포트 전에 설정해야 합니다.
warnings.filterwarnings("ignore", category=UserWarning)

from paddleocr import PaddleOCR


def preprocess_image(img_path):
    """
    OCR 인식률을 높이기 위해 이미지를 보정하는 전처리 함수.
    
    기존의 노이즈를 제거하기 위해 Median Blur를 적용하고,  
    Otsu 알고리즘을 통해 글자와 배경을 가장 깔끔하게 분리합니다.

    :param img_path: 전처리할 원본 이미지의 파일 경로 (str)
    :return: 전처리가 완료된 이미지 데이터 (numpy.ndarray, BGR 포맷) 또는 None
    """
    # 1. 이미지 읽기
    img = cv2.imread(img_path)
    if img is None:
        print(f"오류: 이미지를 읽을 수 없습니다. 경로를 확인하세요: {img_path}")
        return None
    
    # 2. 흑백 변환 (Grayscale) - 차원: (Height, Width)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. 노이즈 제거 (Median Blur)
    # Median Blur는 잡티가 많은 노이즈 제거에 탁월합니다.
    # 커널 크기 5는 노이즈 제거에 적당한 크기입니다.
    denoised = cv2.medianBlur(gray, 5)
    
    # 4. 이진화 (Otsu)
    # Otsu 알고리즘은 이미지의 히스토그램을 분석해 최적의 임계값을 자동으로 찾습니다.
    # 배경이 지저분한 영수증에서 글자만 똑 떼어내는 데 더 효과적일 수 있습니다.
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. 형태학적 연산 (선택 사항: 글자 끊김 보완)
    # 글자가 끊겨 보인다면 커널을 이용해 살짝 이어줍니다. (Closing)
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    
    # 6. PaddleOCR 호환을 위해 1채널(Gray) 이미지를 다시 3채널(BGR)로 변환
    # # (내용은 흑백이지만 데이터 구조만 컬러처럼 맞춤)
    final_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    
    # [디버깅용] 전처리된 전처리 결과 저장
    cv2.imwrite('processed_result.jpg', final_img)
    print(">> 전처리된 이미지를 'processed_result.jpg'로 저장했습니다.")

    return final_img


def main():
    # 이미지 경로
    img_path = './receipt_test.jpg'
    
    print(f"--- 시스템 초기화 중... ---")

    # 1. 모델 초기화
    # use_textline_orientation=True : 글자가 돌아가 있어도 읽어내는 옵션
    ocr = PaddleOCR(use_textline_orientation=True, lang='korean')
        
    print(f"--- '{img_path}' 전처리 및 인식 시작 ---")

    # 2. 이미지 전처리 수행
    processed_img = preprocess_image(img_path)
    
    if processed_img is None:
        return
    
    # 3. 예측 실행 (전처리된 이미지를 넣음)
    # PaddleOCR은 파일 경로뿐만 아니라 numpy array(이미지 데이터)도 직접 받습니다.
    result = ocr.predict(processed_img)

    # 4. 결과 파싱
    if result:
        # v5 모델의 결과 구조 처리
        # 리스트 안에 딕셔너리가 들어있는 구조일 경우
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            data = result[0]
            texts = data.get('rec_texts', [])
            scores = data.get('rec_scores', [])
            
            print(f"\n[총 {len(texts)}개의 텍스트 덩어리를 찾았습니다]\n")

            for i, (text, score) in enumerate(zip(texts, scores)):
                confidence = score * 100
                # 신뢰도 60% 미만은 노이즈일 확률이 높으므로 (Low Confidence) 표시
                if confidence < 60:
                    print(f"[{i+1:02d}] {text} (정확도: {confidence:.2f}%) - [Low Confidence]")
                else:
                    print(f"[{i+1:02d}] {text} (정확도: {confidence:.2f}%)")
        else:
            print("인식된 데이터 구조가 예상과 다릅니다.")
    else:
        print("이미지에서 텍스트를 찾지 못했습니다.")
    
    print("\n--- 인식 종료 ---")

if __name__ == "__main__":
    main()