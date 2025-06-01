import os

def is_polygon(line):
    elements = line.strip().split()
    return (len(elements) - 1) > 4  # class_id 제외한 좌표가 4개(사각형) 초과이면 폴리곤

def detect_bbox_type_in_folder(folder_path):
    rect_count = 0
    poly_count = 0
    total_txt_files = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            total_txt_files += 1
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines:
                    bbox_type = "폴리곤" if is_polygon(line) else "사각형"
                    if bbox_type == "폴리곤":
                        poly_count += 1
                    else:
                        rect_count += 1
                    print(f"이미지: {filename.replace('.txt', '')}, 바운딩 박스 형태: {bbox_type}")

    print()
    print(f"사각형 : {rect_count}/{total_txt_files}")
    print(f"폴리곤 : {poly_count}/{total_txt_files}")

# 사용 예시
detect_bbox_type_in_folder("data_v1/train/square_labels")