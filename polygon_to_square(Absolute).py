import os
import cv2 as cv
import numpy as np
from PIL import Image

# 변환할 데이터셋 폴더 설정
splits = ['valid']  # 필요하면 'valid', 'test'도 추가 가능
src_root = 'data_v1'
dst_root = 'data_v1'  # 결과도 같은 루트에 저장

# def convert_polygon_to_yolo(label_path, image_path):
#     with open(label_path, 'r') as f:
#         lines = f.readlines()

#     img = Image.open(image_path)
#     img_width, img_height = img.size

#     new_lines = []
#     for line in lines:
#         parts = line.strip().split()

#         if len(parts) < 3 or len(parts) % 2 == 0:
#             continue  # 형식이 이상한 라벨은 무시

#         class_id = parts[0]
#         coords = list(map(float, parts[1:]))

#         xs = coords[0::2]
#         ys = coords[1::2]

#         min_x, max_x = min(xs), max(xs)
#         min_y, max_y = min(ys), max(ys)

#         # 중심 좌표와 박스 크기 계산 (정규화 없이)
#         x_center = int((min_x + max_x) / 2)
#         y_center = int((min_y + max_y) / 2)
#         width = int(max_x - min_x)
#         height = int(max_y - min_y)

#         yolo_line = f"{class_id} {x_center} {y_center} {width} {height}"
#         new_lines.append(yolo_line)

#     return new_lines

def convert_polygon_to_yolo(label_path, image_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    img = Image.open(image_path)
    img_width, img_height = img.size

    new_lines = []
    for idx, line in enumerate(lines):
        parts = line.strip().split()

        # (1) 좌표 개수 부족
        if len(parts) < 6:
            print(f"[무시됨 - 좌표 부족] 파일: {os.path.basename(label_path)}, 라인 {idx+1}: {line.strip()}")
            continue

        # (2) 짝수여야 함 (class_id + 좌표쌍 n개 → 홀수)
        if len(parts) % 2 == 0:
            print(f"[무시됨 - 좌표 쌍 아님] 파일: {os.path.basename(label_path)}, 라인 {idx+1}: {line.strip()}")
            continue

        # (3) 사각형 바운딩 박스 (4개 꼭짓점 → 8개 좌표)
        if len(parts) == 9:
            print(f"[무시됨 - 이미 사각형] 파일: {os.path.basename(label_path)}, 라인 {idx+1}")
            continue

        try:
            class_id = parts[0]
            coords = list(map(float, parts[1:]))
        except ValueError:
            print(f"[무시됨 - 숫자 아님] 파일: {os.path.basename(label_path)}, 라인 {idx+1}: {line.strip()}")
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
            print(f"[무시됨 - 너비/높이 0 이하] 파일: {os.path.basename(label_path)}, 라인 {idx+1}: width={width}, height={height}")
            continue

        x_center = int(round((min_x + max_x) / 2))
        y_center = int(round((min_y + max_y) / 2))
        width = int(round(width))
        height = int(round(height))

        yolo_line = f"{class_id} {x_center} {y_center} {width} {height}"
        new_lines.append(yolo_line)

    return new_lines


count = 0
# 데이터셋별 반복 처리
for split in splits:
    src_label_dir = os.path.join(src_root, split, 'before_normal_labels')
    src_image_dir = os.path.join(src_root, split, 'images')
    dst_label_dir = os.path.join(dst_root, split, 'square_labels')
    
    os.makedirs(dst_label_dir, exist_ok=True)

    for filename in os.listdir(src_label_dir):
        if not filename.endswith('.txt'):
            continue

        base_name = os.path.splitext(filename)[0]

        # 이미지 경로 찾기 (.jpg 또는 .png)
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
            print(filename)
            count+=1
        # 변환된 라벨 저장
        dst_label_path = os.path.join(dst_label_dir, filename)
        with open(dst_label_path, 'w') as f:
            for line in yolo_labels:
                f.write(line + '\n')

print("✅ YOLOv11s용 라벨 변환 완료!")
print('변환 오류 개수  :', count)