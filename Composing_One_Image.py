import cv2
import numpy as np
from PIL import Image
import os
import random

# 📂 저장 경로 설정
base_dir = "Making_Train_Data/Compose"
output_image_dir = os.path.join(base_dir, "images")
output_label_dir = os.path.join(base_dir, "labels")

# 폴더 없으면 자동 생성
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

def remove_background(image_path):
    """흰색 배경을 제거하고 투명한 PNG로 변환"""
    image = Image.open(image_path).convert("RGBA")
    datas = image.getdata()

    new_data = []
    for item in datas:
        if item[:3] == (255, 255, 255):  # 흰색 배경 제거
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)

    image.putdata(new_data)
    return image

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

def overlay_image(background, overlay, position, polygon, label, resize_option=False, new_size=(100, 100), output_name="output"):
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

    print(f"📂 이미지 저장 경로: {output_image_path}")
    print(f"📂 라벨 저장 경로: {output_label_path}")

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
        print(f"✅ 이미지 저장 완료: {output_image_path}")
    else:
        print(f"❌ 이미지 저장 실패: {output_image_path}")

    print(f"✅ 라벨 데이터 저장 완료: {output_label_path}")

    # 🚀 OpenCV 방식으로 저장
    img = cv2.imread(output_image_path)

    # 합성 이미지 시각화
    cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

# ✅ 원본 이미지 및 라벨 데이터 로드
Road_Image = "Road_Image/road1.jpg"
Sign_Image = "original_data/images/30km_limit.jpg"

road_image = Image.open(Road_Image)  # 도로 사진
traffic_sign = remove_background(Sign_Image)  # 교통 표지

# ✅ 원본 라벨 파일에서 라벨 값과 폴리곤 좌표 읽기
label_path = Sign_Image.replace("images", "labels").replace(".jpg", ".txt")
label_value, original_polygon = get_label_and_polygon_from_txt(label_path)  # 자동 추출

road_height, road_width, _ = cv2.imread(Road_Image).shape
sign_height, sign_width, _ = cv2.imread(Sign_Image).shape

output_name_road = Road_Image.split('/')[1].split('.')[0]
output_name_sign = Sign_Image.split('/')[2].split('.')[0]
output_name = output_name_road + '_'  + output_name_sign

# ✅ 교통 표지 합성 및 폴리곤 변환 저장 (resize_option=True 적용)
overlay_image(
    road_image,
    traffic_sign,
    position=(random.randint(0, road_height - sign_height), random.randint(0, road_width - sign_width)),
    polygon=original_polygon,
    label=label_value,
    resize_option=True,
    new_size=(150, 150),
    output_name=output_name
)