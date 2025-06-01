import cv2 as cv
import numpy as np
import os
import random

# 회전된 바운딩 박스가 이미지 안에 존재하는지 확인
def is_box_inside_image(bbox, img_w, img_h):
    for x, y in bbox:
        if not (0 <= x < img_w and 0 <= y < img_h):
            return False
    return True

# 폴리곤 → YOLO 형식으로 변환
def polygon_to_yolo(bbox, img_w, img_h):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return x_center, y_center, w, h

# 바운딩 박스를 회전하고 외접 사각형을 구해 YOLO 형식으로 반환
def rotate_bounding_boxes_get_rect(bbox, angle, image_center, img_w, img_h):
    rotated_boxes = []
    rotation_matrix = cv.getRotationMatrix2D(image_center, angle, 1)

    for box in bbox:
        rotated_points = []
        for point in box:
            rotated_point = np.dot(rotation_matrix[:, :2], point) + rotation_matrix[:, 2]
            rotated_points.append(rotated_point)

        rotated_np = np.array(rotated_points, dtype=np.float32)
        rotated_np[:, 0] = np.clip(rotated_np[:, 0], 0, img_w - 1)
        rotated_np[:, 1] = np.clip(rotated_np[:, 1], 0, img_h - 1)

        x, y, w, h = cv.boundingRect(rotated_np.astype(np.int32))
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        rotated_boxes.append((x_center, y_center, w_norm, h_norm))

    return rotated_boxes

# 원본 라벨 파일 복사
def copy_original_label_file(original_label_path, saving_image_path):
    new_file_path = saving_image_path.replace('images', 'labels').replace('.jpg', '.txt')
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    with open(original_label_path, 'r') as original_file:
        lines = original_file.readlines()
    with open(new_file_path, 'w') as new_file:
        new_file.writelines(lines)

# 회전 적용 함수
def apply_rotation(image_root, labels_dir, angle):
    image = cv.imread(image_root)
    if image is None:
        print(f"⚠️ 이미지 로드 실패: {image_root}")
        return

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv.warpAffine(image, M, (w, h))

    label_path = os.path.join(labels_dir, os.path.basename(image_root).replace('.jpg', '.txt'))
    if not os.path.exists(label_path):
        print(f"❌ 회전 실패 - 라벨 파일 없음: {label_path}")
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    yolo_boxes = []

    for line in lines:
        elements = line.strip().split()
        label = elements[0]
        coords = list(map(float, elements[1:]))

        if len(coords) % 2 != 0:
            print(f"⚠️ 폴리곤 좌표 쌍 오류 - {label_path}: {coords}")
            continue

        polygon = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        rotated_yolo = rotate_bounding_boxes_get_rect([polygon], angle, center, w, h)
        x, y, box_w, box_h = rotated_yolo[0]

        if 0 <= x <= 1 and 0 <= y <= 1 and box_w > 0 and box_h > 0:
            yolo_boxes.append((label, x, y, box_w, box_h))

    if len(yolo_boxes) == 0:
        print(f"⚠️ 회전 실패 - 유효한 바운딩 박스 없음: {label_path}, angle={angle}")
        return

    saving_path = image_root.replace('.jpg', f'({angle},0,0,0,0,0).jpg')
    cv.imwrite(saving_path, rotated_image)

    label_file = os.path.join(labels_dir, os.path.basename(saving_path).replace('.jpg', '.txt'))
    with open(label_file, 'w') as f:
        for label, x, y, w_, h_ in yolo_boxes:
            f.write(f"{label} {x:.6f} {y:.6f} {w_:.6f} {h_:.6f}\n")

def apply_brightness(image_root, labels_dir, brightness):
    image = cv.imread(image_root)
    if image is None:
        print(f"⚠️ 밝기 실패 - 이미지 로드 실패: {image_root}")
        return
    bright_image = cv.convertScaleAbs(image, alpha=1, beta=brightness)
    saving_path = image_root.replace('.jpg', f'(0,{brightness},0,0,0,0).jpg')
    label_file = os.path.join(labels_dir, os.path.basename(image_root).replace('.jpg', '.txt'))
    if not os.path.exists(label_file):
        print(f"❌ 밝기 실패 - 라벨 파일 없음: {label_file}")
        return
    cv.imwrite(saving_path, bright_image)
    copy_original_label_file(label_file, saving_path)

def apply_noise(image_root, labels_dir, mean, sigma):
    image = cv.imread(image_root)
    if image is None:
        print(f"⚠️ 노이즈 실패 - 이미지 로드 실패: {image_root}")
        return
    noise = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy_image = cv.add(image, noise)
    saving_path = image_root.replace('.jpg', f'(0,0,0,({mean},{sigma}),0).jpg')
    label_file = os.path.join(labels_dir, os.path.basename(image_root).replace('.jpg', '.txt'))
    if not os.path.exists(label_file):
        print(f"❌ 노이즈 실패 - 라벨 파일 없음: {label_file}")
        return
    cv.imwrite(saving_path, noisy_image)
    copy_original_label_file(label_file, saving_path)

def apply_random_erasing(image_root, labels_dir, x, y, width, height):
    image = cv.imread(image_root)
    if image is None:
        print(f"⚠️ 이레이징 실패 - 이미지 로드 실패: {image_root}")
        return
    image[y:y + height, x:x + width] = (0, 0, 0)
    saving_path = image_root.replace('.jpg', f'(0,0,0,0,({x},{y},{width},{height}),0).jpg')
    label_file = os.path.join(labels_dir, os.path.basename(image_root).replace('.jpg', '.txt'))
    if not os.path.exists(label_file):
        print(f"❌ 이레이징 실패 - 라벨 파일 없음: {label_file}")
        return
    cv.imwrite(saving_path, image)
    copy_original_label_file(label_file, saving_path)

def apply_augmentation(image_root, labels_dir):
    # print('----------------', os.path.basename(image_root), '----------------')
    augmentation = [1, 1, 1, 1]

    # 회전 각도 후보 리스트 생성 및 셔플
    angles = [i for i in range(-45, 46) if i != 0]
    random.shuffle(angles)
    for i in range(augmentation[0]):
        angle = angles.pop()
        apply_rotation(image_root, labels_dir, angle)
        print(f'✅ 회전 {i+1}차 완료 / 회전량 : ', angle)

    # 밝기 조절 후보 리스트 생성 및 셔플
    brightness_values = [i for i in range(-45, 46) if i != 0]
    random.shuffle(brightness_values)
    for i in range(augmentation[1]):
        brightness = brightness_values.pop()
        apply_brightness(image_root, labels_dir, brightness)
        print(f'✅ 밝기 {i+1}차 완료 / 밝기량 : ', brightness)

    # 노이즈 시그마 값 후보 리스트 생성 및 셔플
    sigma_values = [i for i in range(1, 6)]
    random.shuffle(sigma_values)
    for i in range(augmentation[2]):
        sigma = sigma_values.pop()
        mean = 0
        apply_noise(image_root, labels_dir, mean, sigma)
        print(f'✅ 노이즈 {i+1}차 완료 / 노이즈 평균값, 시그마값 : ', mean, sigma)

    # 이레이징 좌표 및 크기 랜덤 (좌표가 겹치더라도 큰 문제는 아님)
    used_coords = set()
    for i in range(augmentation[3]):
        while True:
            x_y = random.randrange(10, 150, 10)
            wh = random.randrange(30, 70, 10)
            if (x_y, wh) not in used_coords:
                used_coords.add((x_y, wh))
                break
        apply_random_erasing(image_root, labels_dir, x_y, x_y, wh, wh)
        print(f'✅ 이레이징 {i+1}차 완료 / 이레이징 좌표 : ', x_y, wh)

# 실행
origin_image = 'data_v1/train/images'
labels_dir = 'data_v1/train/labels'
image_list = [img for img in os.listdir(origin_image) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]

for index, image_file in enumerate(image_list):
    image_path = os.path.join(origin_image, image_file)
    print(f'진행도: {index + 1} / {len(image_list)} ------------------ ', os.path.basename(image_path))
    apply_augmentation(image_path, labels_dir)