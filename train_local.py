import os
import paddle
import subprocess
import sys

# ==========================================
# 🔧 사용자 설정 (여기만 내 환경에 맞게 수정)
# ==========================================
# 1. 사용할 GPU 번호 (0번이 메인)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 2. 데이터 경로 (rec_gt.txt와 crop_img 폴더가 있는 곳)
# 주의: PPOCRLabel 결과물인 crop_img 폴더를 지정해야 함
DATA_DIR = "./train_data/crop_img"
LABEL_FILE = "rec_gt.txt"  # DATA_DIR 안에 들어있어야 함

# 3. 학습 결과 저장 경로
OUTPUT_DIR = "./output/rec_korean_finetune"

# 4. 사전 학습 모델 (Pre-trained Model) 경로
# (없으면 자동으로 다운로드합니다)
PRETRAINED_MODEL_URL = "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_train.tar"
PRETRAINED_MODEL_DIR = "./pretrain_models/korean_PP-OCRv3_rec_train"

def download_model():
    """사전 학습된 한국어 모델 다운로드"""
    if not os.path.exists(PRETRAINED_MODEL_DIR):
        print(f"--- 📥 Pre-trained 모델 다운로드 중... ---")
        os.makedirs("./pretrain_models", exist_ok=True)
        
        # Windows 호환 다운로드 및 압축 해제
        try:
            import urllib.request
            import tarfile
            
            tar_path = "model.tar"
            print(f"다운로드 주소: {PRETRAINED_MODEL_URL}")
            urllib.request.urlretrieve(PRETRAINED_MODEL_URL, tar_path)
            
            print("압축 해제 중...")
            with tarfile.open(tar_path) as tar:
                # 보안 경고 무시하고 풀기 (신뢰할 수 있는 소스임)
                tar.extractall(path="./pretrain_models")
            
            if os.path.exists(tar_path):
                os.remove(tar_path)
            print("--- ✅ 모델 준비 완료 ---")
        except Exception as e:
            print(f"❌ 모델 다운로드 실패: {e}")
            print("수동으로 다운로드하여 ./pretrain_models 폴더에 넣어주세요.")
            sys.exit(1)

def create_config():
    """학습을 위한 yml 설정 파일을 생성합니다."""
    # 데이터셋 경로 절대경로로 변환 (에러 방지)
    abs_data_dir = os.path.abspath(DATA_DIR).replace("\\", "/")
    abs_label_file = f"{abs_data_dir}/{LABEL_FILE}"
    abs_output_dir = os.path.abspath(OUTPUT_DIR).replace("\\", "/")
    abs_pretrained_dir = os.path.abspath(PRETRAINED_MODEL_DIR).replace("\\", "/")

    # YAML 설정 내용 (들여쓰기 중요)
    config_content = f"""
Global:
  use_gpu: true
  epoch_num: 100
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: {abs_output_dir}
  save_epoch_step: 20
  eval_batch_step: [0, 200]
  cal_metric_during_train: true
  pretrained_model: {abs_pretrained_dir}/best_accuracy
  checkpoints:
  save_inference_dir:
  use_visualdl: false
  infer_img: 
  character_dict_path: PaddleOCR/ppocr/utils/dict/korean_dict.txt
  max_text_length: 25
  infer_mode: false
  use_space_char: true
  save_res_path: {abs_output_dir}/predicts_korean.txt

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.001
    warmup_epoch: 5
  regularizer:
    name: 'L2'
    factor: 0.00001

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform:
  Backbone:
    name: MobileNetV1Enhance
    scale: 0.5
  Neck:
    name: SequenceEncoder
    encoder_type: svtr
    dims: 64
    depth: 2
    hidden_dims: 120
    use_guide: True
  Head:
    name: CTCHead
    fc_decay: 0.00001

Loss:
  name: CTCLoss

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: SimpleDataSet
    data_dir: {abs_data_dir}
    label_file_list:
    - {abs_label_file}
    transforms:
    - DecodeImage: 
        img_mode: BGR
        channel_first: false
    - RecAug:
    - CTCLabelEncode: 
    - RecResizeImg:
        image_shape: [3, 48, 320]
    - KeepKeys:
        keep_keys: ['image', 'label', 'length']
  loader:
    shuffle: true
    batch_size_per_card: 16
    drop_last: true
    num_workers: 0

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: {abs_data_dir}
    label_file_list:
    - {abs_label_file}
    transforms:
    - DecodeImage: 
        img_mode: BGR
        channel_first: false
    - CTCLabelEncode: 
    - RecResizeImg:
        image_shape: [3, 48, 320]
    - KeepKeys:
        keep_keys: ['image', 'label', 'length']
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 16
    num_workers: 0
"""
    os.makedirs("configs/rec/custom", exist_ok=True)
    with open("configs/rec/custom/train_local.yml", "w", encoding="utf-8") as f:
        f.write(config_content)
    print(">> 📄 학습 설정 파일 생성 완료: configs/rec/custom/train_local.yml")

def main():
    print("\n" + "="*50)
    print("   🚀 로컬 GPU 학습 준비 시작")
    print("="*50 + "\n")

    # 1. GPU 체크
    try:
        paddle.utils.run_check()
    except Exception:
        print("❌ GPU 체크 실패. 환경 설정을 확인해주세요.")
        return
    
    # 2. 데이터셋 확인
    if not os.path.exists(f"{DATA_DIR}/{LABEL_FILE}"):
        print(f"❌ 오류: 데이터 파일이 없습니다 -> {DATA_DIR}/{LABEL_FILE}")
        print("PPOCRLabel에서 Export한 crop_img 폴더가 맞는지 확인하세요.")
        return

    # 3. 모델 다운로드 및 설정 생성
    download_model()
    create_config()
    
    # 4. PaddleOCR 소스코드(학습 툴) 다운로드
    # pip install paddleocr는 '사용' 도구이고, '학습' 도구(train.py)는 깃허브 소스에 있습니다.
    if not os.path.exists("PaddleOCR/tools/train.py"):
        print("⚠️ 학습용 도구(PaddleOCR Source)가 없습니다. 다운로드합니다...")
        try:
            subprocess.run(["git", "clone", "https://github.com/PaddlePaddle/PaddleOCR.git"], check=True)
            print("--- ✅ PaddleOCR 소스코드 다운로드 완료 ---")
        except Exception as e:
            print(f"❌ Git Clone 실패: {e}")
            print("git이 설치되어 있는지 확인하거나, 수동으로 다운로드해주세요.")
            return

    print("\n" + "="*50)
    print("   🔥 학습 시작! (로그가 올라오면 성공입니다)")
    print("   중단하려면 터미널에서 Ctrl+C를 누르세요.")
    print("="*50 + "\n")
    
    # 5. 학습 실행
    # 파이썬 내부에서 명령어를 호출합니다.
    cmd = [
        sys.executable,  # 현재 파이썬 실행파일 경로
        "PaddleOCR/tools/train.py", 
        "-c", "configs/rec/custom/train_local.yml"
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()