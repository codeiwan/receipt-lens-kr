# receipt-lens-kr (한국어 영수증 OCR)

PaddleOCR 기반의 로컬 구동 한국어 영수증 OCR 프로젝트입니다.  
외부 서버 통신 없이 로컬 환경에서 영수증 이미지의 텍스트를 인식하며, 구겨짐이나 기울어짐 등 현실적인 노이즈에 대응하는 것을 목표로 합니다.

## 📌 주요 특징
- **Local First:** 외부 API 호출 없이 로컬 장비에서 독립적으로 수행
- **High Accuracy:** 한국어, 영어, 숫자 혼용 영수증 인식에 최적화
- **Robust:** 기울기 보정 및 노이즈 대응 (PaddleOCR v5 기반)

## 🛠 환경 요구사항 (Prerequisites)
이 프로젝트는 다음 환경에서 테스트되었습니다.
- **OS:** Windows / Mac / Linux
- **Python:** `3.10.11` (호환성을 위해 3.10 버전을 권장합니다)
- **H/W:** CPU 모드 기본 지원 (GPU 설정 선택 가능)

## 🚀 설치 및 시작하기 (Installation)

### 1. 프로젝트 클론 및 폴더 이동
```bash
git clone [레포지토리 주소]
cd [프로젝트 폴더명]
```

### 2. 가상환경 생성 및 활성화
Python 3.10 환경에서 가상환경을 생성합니다.
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. 의존성 패키지 설치
이 프로젝트는 `pip-tools`로 의존성을 관리합니다. 아래 명령어로 필요한 라이브러리를 설치하세요.
```bash
pip install -r requirements.txt
```

---

## 💻 실행 방법 (Usage)
테스트 이미지를 준비한 후 `main.py`를 실행하여 OCR을 수행합니다.

```bash
python main.py
```

### 💡 최초 실행 시 참고사항
처음 실행 시, PaddleOCR에 필요한 모델 파일(Detection, Recognition, Classifier)이 자동으로 다운로드됩니다.
- 다운로드 용량: 약 100~200MB
- 소요 시간: 네트워크 환경에 따라 1~2분 소요될 수 있습니다.
- 로그에 `Downloading...` 메시지가 표시되니 완료될 때까지 기다려주세요.

---

## 🧑‍💻 개발자 가이드 (Dependency Management)
이 프로젝트는 의존성 관리를 위해 `pip-tools`를 사용합니다. 패키지를 추가하거나 변경할 경우 아래 워크플로우를 따릅니다.

1. **`requirements.in` 수정**
   직접 사용하는 최상위 패키지만 명시합니다.
   ```text
   paddlepaddle
   paddleocr
   ```

2. **requirements.txt 컴파일 (Locking)**
   ```bash
   pip install pip-tools
   pip-compile requirements.in -o requirements.txt
   ```

3. **환경 동기화 (Sync)**
   로컬 환경을 `requirements.txt`와 완벽하게 일치시킵니다. (불필요한 패키지 삭제됨)
   ```bash
   pip-sync requirements.txt
   ```

---

## 🗑️ 제거 및 정리 (Uninstall & Cleanup)
프로젝트를 완전히 삭제하고 싶은 경우, 아래 두 가지 경로의 폴더를 삭제해야 합니다.

1. **프로젝트 폴더 및 가상환경 (`.venv`)**
   - 현재 작업 중인 프로젝트 폴더 전체를 삭제합니다.

2. **모델 캐시 폴더 (`.paddlex`)**
   - PaddleOCR이 다운로드한 모델 파일은 사용자 홈 디렉터리에 저장됩니다. 이 폴더를 삭제해야 디스크 공간이 완전히 확보됩니다.
   - **경로:** `C:\Users\{사용자명}\.paddlex` (Windows 기준)

---

## 📝 License
This project is based on [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).