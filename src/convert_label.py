import os
import cv2
import json
import numpy as np

# ==========================================
# 🔧 경로 설정 (내 환경에 맞게 수정)
# ==========================================
# 1. PPOCRLabel로 만든 라벨 파일 경로
LABEL_FILE = "receipts/Label.txt"

# 2. 원본 이미지가 들어있는 폴더 (Label.txt에 적힌 경로의 상위 폴더)
# 예: Label.txt 안에 'receipts/img.jpg'라고 되어 있으면, 현재 위치에 'receipts' 폴더가 있어야 함
IMAGE_ROOT = "./" 

# 3. 결과물이 저장될 폴더 (자동 생성됨)
OUTPUT_DIR = "./train_data/crop_img"
OUTPUT_GT_FILE = "./train_data/crop_img/rec_gt.txt"

def get_rotate_crop_image(img, points):
    """
    4개의 좌표(points)를 이용하여 이미지를 반듯하게 펴서 자릅니다 (Perspective Transform)
    """
    points = np.array(points, dtype=np.float32)
    
    # 좌상, 우상, 우하, 좌하 순서로 정렬 (대략적)
    # x축 기준 정렬 후, 왼쪽 2개/오른쪽 2개 나눔
    pts_x_sorted = points[np.argsort(points[:, 0]), :]
    left_most = pts_x_sorted[:2, :]
    right_most = pts_x_sorted[2:, :]
    
    # 왼쪽 중 y가 작은게 좌상(tl), 큰게 좌하(bl)
    tl = left_most[np.argsort(left_most[:, 1]), :][0]
    bl = left_most[np.argsort(left_most[:, 1]), :][1]
    
    # 오른쪽 중 y가 작은게 우상(tr), 큰게 우하(br)
    tr = right_most[np.argsort(right_most[:, 1]), :][0]
    br = right_most[np.argsort(right_most[:, 1]), :][1]

    # 변환 후 이미지의 너비/높이 계산
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    width = int(max(w_top, w_bot))
    
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    height = int(max(h_left, h_right))
    
    # 변환 행렬 계산
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # 이미지 자르기 (Warp)
    crop_img = cv2.warpPerspective(img, M, (width, height))
    return crop_img

def main():
    # 저장 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # rec_gt.txt 파일 열기
    with open(OUTPUT_GT_FILE, 'w', encoding='utf-8') as out_f:
        # Label.txt 읽기
        with open(LABEL_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        print(f"총 {len(lines)}장의 이미지를 처리합니다...")
        
        count = 0
        for line in lines:
            try:
                # 탭(\t)으로 이미지 경로와 라벨 데이터 분리
                img_path_str, json_str = line.strip().split('\t', 1)
                
                # 실제 이미지 경로 조합
                full_img_path = os.path.join(IMAGE_ROOT, img_path_str)
                
                # 이미지 로드
                img = cv2.imread(full_img_path)
                if img is None:
                    print(f"❌ 이미지를 찾을 수 없음: {full_img_path}")
                    continue
                
                # 라벨 데이터 파싱
                labels = json.loads(json_str)
                
                # 각 박스별로 크롭 수행
                for i, item in enumerate(labels):
                    points = item['points']
                    text = item['transcription']
                    
                    # 1. 이미지 자르기
                    crop = get_rotate_crop_image(img, points)
                    
                    # 2. 자른 이미지 저장
                    # 파일명: 원본파일명_인덱스.jpg
                    file_base = os.path.basename(img_path_str).split('.')[0]
                    crop_filename = f"{file_base}_{i}.jpg"
                    crop_path = os.path.join(OUTPUT_DIR, crop_filename)
                    
                    cv2.imwrite(crop_path, crop)
                    
                    # 3. 정답지(rec_gt.txt)에 기록
                    # 포맷: 파일명.jpg\t텍스트
                    out_f.write(f"{crop_filename}\t{text}\n")
                    count += 1
                    
            except Exception as e:
                print(f"⚠️ 에러 발생 ({line[:20]}...): {e}")

    print("=" * 50)
    print(f"✅ 변환 완료!")
    print(f"   - 생성된 조각 이미지 수: {count}개")
    print(f"   - 저장 위치: {OUTPUT_DIR}")
    print(f"   - 정답 파일: {OUTPUT_GT_FILE}")
    print("=" * 50)

if __name__ == "__main__":
    main()