import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import random

'''
실행 시 수정해야 할 코드
15번째 줄, output_label_dir = os.path.join(base_dir, "before_normal_labels")
        (before_normal_labels) or (labels)
158번째 줄, label_path = Sign_Image.replace("images", "before_normal_labels").replace(".jpg", ".txt")
        (before_normal_labels) or (labels)
'''

# 📂 저장 경로 설정
base_dir = "data_v8/train"
save_label_file = "before_normal_labels"
output_image_dir = os.path.join(base_dir, "images")
output_label_dir = os.path.join(base_dir, save_label_file)

# 폴더 없으면 자동 생성
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

def remove_background_with_polygon(image_path, polygon):
    """
    특정 폴리곤 내부는 유지하고, 바깥쪽 흰색 배경을 제거하는 함수
    :param image_path: 원본 이미지 경로
    :param polygon: 유지할 영역의 폴리곤 좌표 리스트 [(x1, y1), (x2, y2), ...]
    :return: 배경이 제거된 이미지 (PIL 이미지)
    """
    # 🖼 이미지 불러오기 (RGBA 모드)
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size

    # 🖍 폴리곤 마스크 생성 (흑백)
    mask = Image.new("L", (width, height), 0)  # 전체 투명한 마스크 생성
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon, fill=255)  # 폴리곤 내부를 불투명(255)으로 채움

    # 🛑 NumPy 배열 변환 (이미지 및 마스크)
    image_data = np.array(image)  # (H, W, 4) RGBA
    mask_data = np.array(mask)  # (H, W) 0~255 값 (0=투명, 255=불투명)

    # ✅ 흰색 배경을 투명하게 처리
    r, g, b, a = image_data[:, :, 0], image_data[:, :, 1], image_data[:, :, 2], image_data[:, :, 3]

    # 흰색 배경 탐색 (약간의 허용 범위 포함)
    white_mask = (r > 200) & (g > 200) & (b > 200)  # 밝은 흰색 계열 찾기

    # 흰색 배경이면서 폴리곤 바깥쪽이면 투명 처리
    a[white_mask & (mask_data == 0)] = 0

    # 🖼 최종 이미지 생성 (PIL 변환)
    result_image = Image.fromarray(image_data)

    return result_image

def transform_polygon_coords(polygon, old_size, new_size, position):
    """
    폴리곤 좌표를 리사이징하고 도로 사진 기준으로 변환
    :param polygon: 원본 폴리곤 좌표 리스트 [(x1, y1), (x2, y2), ...]
    :param old_size: 원본 이미지 크기 (w, h)
    :param new_size: 리사이징된 크기 (w, h)
    :param position: 도로 사진에서의 위치 (x_offset, y_offset)
    :return: 변환된 폴리곤 좌표 리스트
    """
    w_ratio = new_size[0] / old_size[0]
    h_ratio = new_size[1] / old_size[1]

    transformed_polygon = [
        (int(x * w_ratio + position[0]), int(y * h_ratio + position[1])) for x, y in polygon
    ]

    return transformed_polygon

def get_label_and_polygon_from_txt(label_path):
    """라벨 파일에서 첫 번째 값을 라벨로 사용하고, 나머지 값을 폴리곤 좌표로 변환"""
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    label = lines[0].strip()[0]  # 첫 번째 줄: 라벨 값 (숫자)
    
    # 두 번째 줄에 좌표들이 존재한다고 가정하고 처리
    coordinates = lines[0].strip().split()[1:]  # 좌표들이 공백으로 구분됨
    polygon = [tuple([int(coordinates[i]), int(coordinates[i+1])]) for i in range(0, len(coordinates), 2)]

    return label, polygon  # (라벨 값, 폴리곤 좌표 리스트) 반환

def overlay_image(background, overlay, position, polygon, label, new_size, resize_option, visuality, printing, output_name="output"):
    """
    도로 이미지 위에 교통 표지를 합성하고, 변환된 바운딩 박스 좌표를 저장
    :param background: 도로 사진 (PIL 이미지)
    :param overlay: 교통 표지 (PIL 이미지)
    :param position: 도로 사진에서의 배치 위치 (x, y)
    :param polygon: 원본 교통 표지의 폴리곤 좌표 리스트
    :param label: 교통 표지 라벨
    :param resize_option: 리사이징 여부
    :param new_size: 새로운 크기
    :param output_name: 저장할 파일 이름 (확장자 제외)
    """
    print(f"🔹 삽입 위치: {position}")
    old_size = overlay.size  # 원본 크기
    if resize_option:
        overlay = overlay.resize(new_size, Image.Resampling.LANCZOS)  # 리사이징

    # 도로 이미지에 합성 (투명 배경 유지)
    background.paste(overlay, position, overlay)

    # 변환된 폴리곤 좌표 계산
    transformed_polygon = transform_polygon_coords(polygon, old_size, new_size if resize_option else old_size, position)

    # 📂 저장 경로 설정
    output_image_path = os.path.join(output_image_dir, f"{output_name}.jpg")
    output_label_path = os.path.join(output_label_dir, f"{output_name}.txt")

    # 저장된 파일이 실제로 존재하는지 다시 확인
    if os.path.exists(output_image_path):
        print(f"🎯 확인: 이미지가 존재합니다: {output_image_path}")
    else:
        print(f"🚨 확인: 이미지가 저장되지 않았습니다: {output_image_path}")

    # # 저장 경로 확인
    # print(f"📂 이미지 저장 경로: {output_image_path}")
    # print(f"📂 라벨 저장 경로: {output_label_path}")

    # 🚀 변환된 좌표를 TXT 형식으로 저장
    with open(output_label_path, "w") as f:
        f.write(f"{label} ")  # 첫 줄: 라벨 값 (숫자)
        f.write(" ".join([f"{x} {y}" for x, y in transformed_polygon]))  # 두 번째 줄: 폴리곤 좌표

    # 🖼 OpenCV에서 읽기 위해 PIL 이미지를 NumPy 배열로 변환
    opencv_image = np.array(background)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

    # 합성 이미지 저장
    cv2.imwrite(output_image_path, opencv_image)
    # ✅ 저장 확인
    if os.path.exists(output_image_path):
        if printing:
            print(f"✅ 이미지 저장 완료: {output_image_path}")
    else:
        print(f"❌ 이미지 저장 실패: {output_image_path}")

    if printing:
        print(f"✅ 라벨 데이터 저장 완료: {output_label_path}")

    if visuality:
        # 🚀 OpenCV 방식으로 저장
        img = cv2.imread(output_image_path)

        # 합성 이미지 시각화
        cv2.imshow('image', img)
        cv2.waitKey()
        cv2.destroyAllWindows()

def Preparing(road_image, sign_image):
    # ✅ 원본 이미지 및 라벨 데이터 로드
    Road_Image = road_image
    Sign_Image = sign_image

    # ✅ 원본 라벨 파일에서 라벨 값과 폴리곤 좌표 읽기
    label_path = Sign_Image.replace("images", save_label_file).replace(".jpg", ".txt")
    label_value, original_polygon = get_label_and_polygon_from_txt(label_path)  # 자동 추출

    road_image = Image.open(Road_Image)  # 도로 사진
    traffic_sign = remove_background_with_polygon(Sign_Image, original_polygon)  # 교통 표지

    road_height, road_width, _ = cv2.imread(Road_Image).shape
    sign_height, sign_width, _ = cv2.imread(Sign_Image).shape

    # 합성할 이미지의 위치
    if road_width - sign_width > 0:
        max_width = road_width - sign_width
    else:
        max_width = 0
    
    if road_height - sign_height > 0:
        max_height = road_height - sign_height
    else:
        max_height = 0

    output_name_road = Road_Image.split('/')[1].split('.')[0]
    # output_name_sign = Sign_Image.split('/')[2].split('.')[0]
    for name_sign in Sign_Image.split('/'):
        if '.jpg' in name_sign:
            output_name_sign = name_sign.split('.')[0]
    output_name = output_name_road + '_'  + output_name_sign


    # ✅ 교통 표지 합성 및 폴리곤 변환 저장 (resize_option=True 적용)
    overlay_image(
        road_image,
        traffic_sign,
        position=(random.randint(0, max_width), random.randint(0, max_height)),
        polygon=original_polygon,
        label=label_value,
        new_size=(int(road_width/10)*2, int(road_height/10)*2),
        resize_option=True,
        visuality = False,
        printing = False,
        output_name=output_name
    )

road_folder = 'Road_Image'
sign_folder = 'data_v4/train/images'
for road in os.listdir(road_folder):
    for sign in os.listdir(sign_folder):
        if (sign == '.DS_Store') or (road == '.DS_Store'):
            continue
        
        road_root = os.path.join(road_folder, road)
        sign_root = os.path.join(sign_folder, sign)
        # print(road_root, sign_root)
        Preparing(road_root, sign_root)