import os
from PIL import Image

# 변환할 데이터셋 폴더 설정
splits = ['train']  # 'train', 'test' 등을 여기에 추가 가능
src_root = 'data_v1'
dst_root = 'data_v1'  # 결과도 같은 루트에 저장

def convert_polygon_or_keep_yolo(label_path, image_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    img = Image.open(image_path)
    img_width, img_height = img.size

    new_lines = []
    for line in lines:
        parts = line.strip().split()

        if len(parts) == 5:
            # 이미 YOLO 형식 (x_center, y_center, w, h 정규화된 사각형)
            try:
                float_values = list(map(float, parts[1:]))  # 검증용
                yolo_line = " ".join(parts)
                new_lines.append(yolo_line)
            except ValueError:
                continue  # 숫자 변환 실패 시 무시

        elif len(parts) >= 7 and len(parts) % 2 == 1:
            # Polygon 형식 (정규화된 좌표들)
            class_id = parts[0]
            try:
                coords = list(map(float, parts[1:]))
            except ValueError:
                continue

            xs = coords[0::2]
            ys = coords[1::2]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # 중심 좌표와 박스 크기 계산
            x_center = (min_x + max_x) / 2
            y_center = (min_y + max_y) / 2
            width = max_x - min_x
            height = max_y - min_y

            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            new_lines.append(yolo_line)
        else:
            # 잘못된 형식은 무시
            continue

    return new_lines

# 데이터셋별 반복 처리
for split in splits:
    src_label_dir = os.path.join(src_root, split, 'normal_labels')
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
        yolo_labels = convert_polygon_or_keep_yolo(label_path, image_path)

        # 변환된 라벨 저장
        dst_label_path = os.path.join(dst_label_dir, filename)
        with open(dst_label_path, 'w') as f:
            for line in yolo_labels:
                f.write(line + '\n')

print("✅ YOLO 및 Polygon 라벨 변환 완료!")
