import os
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

# ==========================================
# 🔧 사용자 설정 (경로 확인 필수)
# ==========================================
# 1. 학습된 모델 경로 (User가 강조한 'output' 폴더 사용)
# best_model 폴더 안에 .pdparams, .pdopt 등이 들어있어야 함
REC_MODEL_DIR = "./output/rec_korean_finetune/best_model"

# 2. 사전 파일 경로 (학습 때 썼던 것과 똑같아야 함)
DICT_PATH = "./PaddleOCR/ppocr/utils/dict/korean_dict.txt"

# 3. 테스트할 이미지 경로
# (기존에 있던 테스트용 이미지나, 아무 영수증 이미지나 지정하세요)
TEST_IMAGE_PATH = "./receipts/receipt_test.jpg" 

# 4. 결과 이미지 저장 경로
RESULT_SAVE_PATH = "./result_inference.jpg"

def main():
    print(f"--- 🚀 추론 시작: {REC_MODEL_DIR} 모델 사용 ---")

    # 1. 파일 존재 여부 확인 (안전장치)
    if not os.path.exists(REC_MODEL_DIR):
        print(f"❌ 오류: 모델 폴더를 찾을 수 없습니다 -> {REC_MODEL_DIR}")
        return
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ 오류: 테스트 이미지가 없습니다 -> {TEST_IMAGE_PATH}")
        print("   코드 상단의 TEST_IMAGE_PATH를 실제 이미지 경로로 수정해주세요.")
        return

    # 2. PaddleOCR 엔진 초기화
    # - det_model_dir: 지정 안 하면 기본(Pretrained) 모델 자동 다운로드 (위치 찾기용)
    # - rec_model_dir: 우리가 방금 학습시킨 모델 (글자 읽기용)
    # - rec_char_dict_path: 한국어 사전
    ocr = PaddleOCR(
        use_angle_cls=True,         # 문자가 뒤집혀도 인식하도록 설정
        lang='korean',              # 기본 언어 설정
        rec_model_dir=REC_MODEL_DIR,      # ★ 핵심: 내 모델 사용
        rec_char_dict_path=DICT_PATH,     # ★ 핵심: 내 사전 사용
        use_gpu=True,               # GPU 사용
        show_log=False              # 자잘한 로그 숨김
    )

    print("--- 📸 이미지 분석 중... ---")
    
    # 3. OCR 실행
    # cls=True: 방향 분류기 사용 (뒤집힌 글자 바로잡기)
    result = ocr.ocr(TEST_IMAGE_PATH, cls=True)

    # 4. 결과 출력
    print("\n" + "="*40)
    print("   🧾 인식 결과")
    print("="*40)
    
    if result and result[0]:
        boxes = []
        txts = []
        scores = []
        
        for idx, line in enumerate(result[0]):
            box = line[0]           # 좌표
            txt = line[1][0]        # 인식된 글자
            score = line[1][1]      # 확신도 (0~1)
            
            boxes.append(box)
            txts.append(txt)
            scores.append(score)
            
            # 한 줄씩 출력
            print(f"[{idx+1:02d}] {txt} \t(확신도: {score:.4f})")
        
        print("="*40 + "\n")

        # 5. 시각화 (이미지에 박스 그리고 저장)
        # 폰트 경로: 윈도우 기본 맑은고딕 사용 (없으면 기본 폰트)
        font_path = "C:/Windows/Fonts/malgun.ttf"
        if not os.path.exists(font_path):
            font_path = "./PaddleOCR/doc/fonts/korean.ttf" # 대체 폰트

        try:
            image = Image.open(TEST_IMAGE_PATH).convert('RGB')
            im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
            im_show = Image.fromarray(im_show)
            im_show.save(RESULT_SAVE_PATH)
            print(f"✅ 결과 이미지 저장 완료: {RESULT_SAVE_PATH}")
        except Exception as e:
            print(f"⚠️ 이미지 저장 중 오류 발생 (결과는 텍스트로 확인하세요): {e}")

    else:
        print("❌ 글자를 찾지 못했습니다.")

if __name__ == "__main__":
    main()