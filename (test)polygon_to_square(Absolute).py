import os
import cv2 as cv
import numpy as np
from PIL import Image

# 변환할 데이터셋 폴더 설정
splits = ['valid']  # 필요 시 'train', 'test' 등 추가
src_root = 'data_v1'
dst_root = 'data_v1'  # 결과도 같은 루트에 저장

def convert_polygon_to_yolo(label_path, image_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    img = Image.open(image_path)
    img_width, img_height = img.size

    new_lines = []
    for idx, line in enumerate(lines):
        parts = line.strip().split()

        # 최소 조건: class_id + 최소 2쌍 이상의 좌표 (즉, 총 5개 이상)
        if len(parts) < 5 or (len(parts) - 1) % 2 != 0:
            print(f"[무시됨 - 형식 오류] 파일: {os.path.basename(label_path)}, 라인 {idx+1}: {line.strip()}")
            continue

        try:
            class_id = parts[0]
            coords = list(map(float, parts[1:]))
        except ValueError:
            print(f"[무시됨 - 숫자 아님] 파일: {os.path.basename(label_path)}, 라인 {idx+1}")
            continue

        xs = coords[0::2]
        ys = coords[1::2]

        if not xs or not ys:
            print(f"[무시됨 - 좌표 없음] 파일: {os.path.basename(label_path)}, 라인 {idx+1}")
            continue

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        width = max_x - min_x
        height = max_y - min_y

        if width <= 0 or height <= 0:
            print(f"[무시됨 - 잘못된 크기] 파일: {os.path.basename(label_path)}, 라인 {idx+1}")
            continue

        # 정규화된 중심 좌표 및 크기
        x_center = ((min_x + max_x) / 2) / img_width
        y_center = ((min_y + max_y) / 2) / img_height
        norm_width = width / img_width
        norm_height = height / img_height

        # 정규화된 YOLO 포맷 (소수점 유지)
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
        new_lines.append(yolo_line)

    return new_lines


# 전체 변환 루프
count = 0  # 변환 실패 개수
for split in splits:
    src_label_dir = os.path.join(src_root, split, 'before_normal_labels')
    src_image_dir = os.path.join(src_root, split, 'images')
    dst_label_dir = os.path.join(dst_root, split, 'square_labels')

    os.makedirs(dst_label_dir, exist_ok=True)

    for filename in os.listdir(src_label_dir):
        if not filename.endswith('.txt'):
            continue

        base_name = os.path.splitext(filename)[0]

        # 이미지 확장자 찾기
        image_path = None
        for ext in ['.jpg', '.png']:
            temp_path = os.path.join(src_image_dir, base_name + ext)
            if os.path.exists(temp_path):
                image_path = temp_path
                break

        if image_path is None:
            print(f"[경고] 이미지 없음: {base_name}")
            continue

        label_path = os.path.join(src_label_dir, filename)
        yolo_labels = convert_polygon_to_yolo(label_path, image_path)

        if not yolo_labels:
            print(f"[스킵됨] 변환된 라벨 없음: {filename}")
            count += 1
            continue

        # 결과 저장
        dst_label_path = os.path.join(dst_label_dir, filename)
        with open(dst_label_path, 'w') as f:
            for line in yolo_labels:
                f.write(line + '\n')

print("\n✅ YOLOv11s용 라벨 변환 완료!")
print(f'⚠️ 변환 실패 파일 수 : {count}')