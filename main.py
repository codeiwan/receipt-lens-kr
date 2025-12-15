from paddleocr import PaddleOCR

# [변경점 1] use_angle_cls -> use_textline_orientation 로 변경
# 최신 버전에서는 이 파라미터 이름을 사용합니다. (기울기 보정 기능)
ocr = PaddleOCR(use_textline_orientation=True, lang='korean') 

# 이미지 경로 지정 (본인이 준비한 이미지 파일명으로 변경)
img_path = './receipt_test.jpg'

print("--- 인식 시작 ---")

# [변경점 2] cls=True 삭제
# 초기화 할 때 이미 기울기 보정(orientation)을 켰으므로, 여기서는 뺍니다.
result = ocr.ocr(img_path)

# 결과 출력
# 결과값이 없으면(인식 실패 시) 에러가 날 수 있으므로 예외처리 추가
if not result or result[0] is None:
    print("텍스트를 찾지 못했습니다.")
else:
    for idx in range(len(result)):
        res = result[idx]
        if res: # 결과가 있을 때만 출력
            for line in res:
                print(f"인식된 텍스트: {line[1][0]} (정확도: {line[1][1]:.4f})")