# receipt-lens-kr (한국어 영수증 OCR)

PaddleOCR 기반의 로컬 구동 한국어 영수증 OCR 프로젝트입니다.  
외부 서버 통신 없이 로컬 환경에서 영수증 이미지의 텍스트를 인식하며, OpenCV 기반의 전처리 파이프라인을 통해 현실적인 노이즈(그림자, 구겨짐)에 대응합니다.

## 📌 주요 특징 (Features)
- **Local First:** 외부 API 호출 없이 로컬 장비(CPU/GPU)에서 독립적으로 수행
- **Image Preprocessing:** OpenCV를 활용한 노이즈 제거(Median Blur) 및 자동 이진화(Otsu Thresholding) 적용
- **Robustness:** 흐릿하거나 구겨진 영수증 이미지에서도 글자 객체를 선명하게 분리

## 🛠 환경 요구사항 (Prerequisites)
이 프로젝트는 다음 환경에서 테스트되었습니다.
- **OS:** Windows / Mac / Linux
- **Python:** `3.10` (권장)
- **Library:** PaddleOCR v2.9+, OpenCV-Python
- **Hardware:** NVIDIA GPU 권장 (학습 시 필수)

## 🚀 설치 및 시작하기 (Installation)

### 1. 프로젝트 클론 및 폴더 이동
```bash
git clone [레포지토리 주소]
cd [프로젝트 폴더명]
```

### 2. 가상환경 생성 및 활성화
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. 의존성 패키지 설치
`pip-tools`를 사용하여 의존성을 관리합니다.
```bash
pip install pip-tools
pip-sync requirements.txt
# 또는 pip install -r requirements.txt
```

---

## 💻 실행 방법 (Usage)

### 1. OCR 인식 실행 (Inference)
테스트할 영수증 이미지를 `receipt_test.jpg`로 저장한 후 아래 명령어를 실행합니다.
```bash
python main.py
```

### 2. 데이터 라벨링 도구 실행 (Development)
학습 데이터를 생성하기 위한 라벨링 툴(PPOCRLabel)은 별도의 스크립트로 실행합니다.
(Windows 환경에서의 라이브러리 충돌 문제를 해결한 스크립트입니다.)

**사전 준비:**
```bash
# 라벨링 툴 의존성 설치 (최초 1회)
pip install PPOCRLabel "paddlex[ocr]"
```

**실행:**
```bash
python run_label.py
```

---

## ⚙️ 동작 과정
1. **전처리 (Preprocessing):** 원본 이미지를 흑백으로 변환하고 노이즈를 제거한 뒤 `processed_result.jpg`로 저장합니다.
2. **인식 (OCR):** 전처리된 이미지를 PaddleOCR 엔진(PP-OCRv5)에 입력하여 텍스트를 추출합니다.
3. **결과 출력:** 인식된 텍스트와 정확도(Confidence)를 터미널에 출력합니다.

---

## 🧑‍💻 개발자 가이드 (Dependency Management)
패키지를 추가하거나 변경할 경우 `requirements.in`을 수정하고 아래 워크플로우를 따릅니다.

1. **`requirements.in` 작성**
   ```text
   paddlepaddle-gpu  # GPU 사용 시 (CPU만 쓴다면 paddlepaddle)
   paddleocr
   opencv-python
   ```

2. **requirements.txt 컴파일**
   ```bash
   pip-compile requirements.in -o requirements.txt
   ```

3. **환경 동기화**
   ```bash
   pip-sync requirements.txt
   ```

---

## 🗑️ 제거 및 정리 (Uninstall & Cleanup)
프로젝트 삭제 시 다음 폴더들을 제거해야 합니다.
1. `.venv` (가상환경)
2. `C:\Users\{사용자명}\.paddlex` (OCR 모델 캐시)
3. `C:\Users\{사용자명}\.paddleocr` (라벨링 툴 모델 캐시)

---

## 📝 License
This project is based on [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).