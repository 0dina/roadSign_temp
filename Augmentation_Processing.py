import cv2 as cv
import numpy as np
import os
import csv
import random
'''
실행 전 필독 사항
바운딩 박스 형식 : 사각형(x, y, w, h)
회전 증강을 통해 바운딩 박스 좌표가 수정되는데 이때 폴리곤 값인 데이터도 처리할 수 있는 코드임
'''
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
        # box: [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        rotated_points = []
        for point in box:
            rotated_point = np.dot(rotation_matrix[:, :2], point) + rotation_matrix[:, 2]
            rotated_points.append(rotated_point)

        # numpy array로 변환 후 좌표 클리핑
        rotated_np = np.array(rotated_points, dtype=np.float32)
        rotated_np[:, 0] = np.clip(rotated_np[:, 0], 0, img_w - 1)
        rotated_np[:, 1] = np.clip(rotated_np[:, 1], 0, img_h - 1)

        # 외접 사각형 계산
        x, y, w, h = cv.boundingRect(rotated_np.astype(np.int32))

        # YOLO 형식으로 정규화
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        rotated_boxes.append((x_center, y_center, w_norm, h_norm))

    return rotated_boxes


# 바운딩 박스를 YOLO 형식으로 저장
def save_bounding_boxes(image_path, bounding_boxes, labels_dir, image_size):
    label_file = os.path.join(labels_dir, os.path.basename(image_path).replace('.jpg', '.txt'))
    img_w, img_h = image_size
    with open(label_file, 'w') as f:
        for label, bbox in bounding_boxes:
            if len(bbox) == 0:
                continue
            x, y, w, h = polygon_to_yolo(bbox, img_w, img_h)
            f.write(f"{label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    # print("YOLO 포맷 바운딩 박스 저장 완료:", label_file)

# 원본 라벨 파일 복사
def copy_original_label_file(original_label_path, saving_image_path, augment):
    new_file_path = saving_image_path.replace('images', 'labels').replace('.jpg', '.txt')
    new_folder = os.path.dirname(new_file_path)
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    with open(original_label_path, 'r') as original_file:
        lines = original_file.readlines()
    with open(new_file_path, 'w') as new_file:
        new_file.writelines(lines)
    # print("원본 라벨 복사 완료")

# 회전 적용 함수
def apply_rotation(image_root, labels_dir, angle):
    image = cv.imread(image_root)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # 회전 적용
    M = cv.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv.warpAffine(image, M, (w, h))

    # 원본 라벨 불러오기
    original_label_file = os.path.join(labels_dir, os.path.basename(image_root).replace('.jpg', '.txt'))
    with open(original_label_file, 'r') as f:
        lines = f.readlines()

    yolo_boxes = []

    for line in lines:
        elements = line.strip().split()
        label = elements[0]
        coords = list(map(float, elements[1:]))

        if len(coords) % 2 != 0:
            print(f"⚠️ 폴리곤 좌표 쌍이 맞지 않음: {coords}")
            continue

        # 폴리곤 점 리스트로 변환
        polygon = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]

        # 회전 → 외접 사각형 → YOLO 좌표 변환
        rotated_yolo = rotate_bounding_boxes_get_rect([polygon], angle, center, w, h)
        x, y, box_w, box_h = rotated_yolo[0]

        if 0 <= x <= 1 and 0 <= y <= 1 and box_w > 0 and box_h > 0:
            yolo_boxes.append((label, x, y, box_w, box_h))

    # 저장 경로 생성
    label_file = os.path.join(labels_dir, os.path.basename(image_root).replace('.jpg', '.txt'))
    saving_path = image_root.replace('.jpg', f'({angle},0,0,0,0,0).jpg')

    # 이미지 저장
    cv.imwrite(saving_path, rotated_image)

    # 라벨 저장
    label_file = os.path.join(labels_dir, os.path.basename(saving_path).replace('.jpg', '.txt'))
    with open(label_file, 'w') as f:
        for label, x, y, w_, h_ in yolo_boxes:
            f.write(f"{label} {x:.6f} {y:.6f} {w_:.6f} {h_:.6f}\n")

    # print(f"(회전)이미지 저장 완료: {saving_path}")


# 밝기 조절
def apply_brightness(image_root, labels_dir, brightness):
    image = cv.imread(image_root)
    bright_image = cv.convertScaleAbs(image, alpha=1, beta=brightness)

    label_file = os.path.join(labels_dir, os.path.basename(image_root).replace('.jpg', '.txt'))
    saving_path = image_root.replace('.jpg', f'(0,{brightness},0,0,0,0).jpg')

    cv.imwrite(saving_path, bright_image)
    copy_original_label_file(label_file, saving_path, 1)
    # print("(밝기)이미지 저장 완료")
    return saving_path

# 노이즈 추가
def apply_noise(image_root, labels_dir, mean, sigma):
    image = cv.imread(image_root)
    noise = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy_image = cv.add(image, noise)

    label_file = os.path.join(labels_dir, os.path.basename(image_root).replace('.jpg', '.txt'))
    saving_path = image_root.replace('.jpg', f'(0,0,0,({mean},{sigma}),0).jpg')
    cv.imwrite(saving_path, noisy_image)
    copy_original_label_file(label_file, saving_path, 0)
    # print("(노이즈)이미지 저장 완료")
    return saving_path

# 랜덤 이레이징
def apply_random_erasing(image_root, labels_dir, x, y, width, height):
    image = cv.imread(image_root)
    image[y:y + height, x:x + width] = (0, 0, 0)

    label_file = os.path.join(labels_dir, os.path.basename(image_root).replace('.jpg', '.txt'))
    saving_path = image_root.replace('.jpg', f'(0,0,0,0,({x},{y},{width},{height}),0).jpg')
    cv.imwrite(saving_path, image)
    copy_original_label_file(label_file, saving_path, 2)
    # print("(이레이징)이미지 저장 완료")
    return saving_path

# 전체 증강 함수
def apply_augmentation(image_root, labels_dir):
    print('-------------------------------------', image_root, '-------------------------------------')
    angle1 = random.randint(-45, 45)
    apply_rotation(image_root, labels_dir, angle1)
    print(image_root, '1차 회전 증강 완료')
    angle2 = random.randint(-45, 45)
    apply_rotation(image_root, labels_dir, angle2)
    print(image_root, '2차 회전 증강 완료')
    angle3 = random.randint(-45, 45)
    apply_rotation(image_root, labels_dir, angle3)
    print(image_root, '3차 회전 증강 완료')

    brightness1 = random.randint(-45, 45)
    apply_brightness(image_root, labels_dir, brightness1)
    print(image_root, '1차 밝기 증강 완료')
    brightness2 = random.randint(-45, 45)
    apply_brightness(image_root, labels_dir, brightness2)
    print(image_root, '2차 밝기 증강 완료')

    mean1, sigma1 = 0, random.randint(0, 5)
    apply_noise(image_root, labels_dir, mean1, sigma1)
    print(image_root, '1차 노이즈 증강 완료')
    mean2, sigma2 = 0, random.randint(0, 5)
    apply_noise(image_root, labels_dir, mean2, sigma2)
    print(image_root, '2차 노이즈 증강 완료')

    x_y_1 = random.randrange(0, 150, 10)
    width_height_1 = random.randrange(30, 70, 10)
    x, y, width, height = x_y_1, x_y_1, width_height_1, width_height_1
    apply_random_erasing(image_root, labels_dir, x, y, width, height)
    print(image_root, '1차 이레이징 증강 완료')
    x_y_2 = random.randrange(0, 150, 10)
    width_height_2 = random.randrange(30, 70, 10)
    x, y, width, height = x_y_2, x_y_2, width_height_2, width_height_2
    apply_random_erasing(image_root, labels_dir, x, y, width, height)
    print(image_root, '2차 이레이징 증강 완료')

# 실행
origin_image = 'data_v1/train/images'
labels_dir = 'data_v1/train/labels'
count = [image for image in os.listdir(origin_image) if image != '.DS_Store']

for index, image_file in enumerate(count):
    print('진행도 ', index+1, ' / ', len(count))
    if image_file.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(origin_image, image_file)
        apply_augmentation(image_path, labels_dir)