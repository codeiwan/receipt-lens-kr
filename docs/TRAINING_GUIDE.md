# 🏋️‍♂️ 모델 학습 가이드 (Training Guide)

이 문서는 PaddleOCR을 로컬 GPU(특히 NVIDIA RTX 40 시리즈) 환경에서 학습시키기 위한 **상세 설정 가이드**입니다.
Python 버전 호환성 및 DLL 누락 문제를 해결한 검증된 절차입니다.

## 🛠️ 1. 사전 요구 사항 (Prerequisites)

* **OS:** Windows 10/11
* **Python:** 3.8 ~ 3.10 (3.10 권장)
* **GPU Driver:** 최신 NVIDIA 드라이버 설치
* **Git:** 설치 필요

## ⚙️ 2. 환경 설정 (Setup Environment)

단순 `pip install`로는 실행되지 않습니다. 아래 순서를 반드시 지켜주세요.

### 2-1. 의존성 설치 (pip-tools 활용)
학습용 라이브러리는 버전 충돌 방지를 위해 버전이 고정(`requirements/train.txt`)되어 있습니다.

```bash
# 1. pip-tools 설치
pip install pip-tools

# 2. 의존성 컴파일 (train.in -> train.txt)
pip-compile requirements/train.in -o requirements/train.txt

# 3. 패키지 동기화 및 설치
pip-sync requirements/train.txt
```

### 2-2. 수동 라이브러리 패치 (DLL Surgery) 💉
PaddlePaddle이 RTX 40 환경에서 `cudnn` 라이브러리를 자동으로 찾지 못하는 문제를 해결해야 합니다.

1.  **cuDNN v8.9.x 다운로드:**
    * [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive) 접속
    * `Download cuDNN v8.9.7 (for CUDA 12.x)` -> `Windows (Zip)` 다운로드
2.  **파일 복사:**
    * 압축 해제 후 `bin` 폴더 안의 **모든 `*.dll` 파일**을 복사합니다.
    * 현재 가상환경의 `Scripts` 폴더 (예: `.venv/Scripts`) 또는 `site-packages/paddle/libs` 폴더에 붙여넣습니다.

### 2-3. 설치 확인
터미널에서 아래 명령어를 입력했을 때, `PaddlePaddle works well on 1 GPU` 메시지가 떠야 합니다.

```bash
python -c "import paddle; paddle.utils.run_check()"
```

---

## 📂 3. 데이터 준비 (Data Preparation)

PPOCRLabel 등을 이용해 라벨링한 데이터를 학습 포맷으로 변환합니다.

1.  **데이터 위치:**
    * 원본 이미지와 라벨링 결과 파일(`Label.txt`)을 `data/raw/` 폴더에 위치시킵니다.
    * 예: `data/raw/receipts/이미지들...` 및 `data/raw/Label.txt`
2.  **전처리 실행:**
    스크립트가 이미지를 글자 단위로 자르고(Crop) 정답지(`rec_gt.txt`)를 생성합니다.

```bash
# 스크립트 실행 (src 폴더 내에 위치)
python src/convert_label.py
```
* 결과물은 `data/processed/` 폴더에 생성됩니다.

---

## 🔥 4. 학습 실행 (Training)

모든 설정이 완료되었습니다. 학습 스크립트를 실행합니다.
이 스크립트는 복잡한 YAML 설정을 파이썬 코드(`train_local.py`) 내에서 자동으로 처리합니다.

```bash
python train_local.py
```

### 주요 확인 사항
* **로그 확인:** 터미널에 `epoch: [1/100] ... loss: ...` 로그가 올라오면 정상입니다.
* **초반 경고(Warning):** `The pretrained params ... not in model` 경고는 정상입니다. (기존 모델의 분류기를 한국어용으로 교체하는 과정입니다.)
* **모델 저장:** 학습된 모델은 `models/finetuned/` 폴더에 저장됩니다.