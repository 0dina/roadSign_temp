import os
import cv2

data_set = 'valid'
# 이미지 파일 경로와 정규화된 라벨 파일 경로
image_folder = "data_v1/" + data_set + "/images"  # 이미지 폴더 경로
normalized_label_folder = "data_v1/" + data_set + "/normal_labels"  # 정규화된 라벨 파일이 저장된 폴더

# 변환된 좌표를 저장할 폴더 (없으면 생성)
absolute_label_folder = "data_v1/" + data_set + "/before_normal_labels"
os.makedirs(absolute_label_folder, exist_ok=True)

# 이미지 파일 리스트 가져오기
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

# 파일 하나씩 처리
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    label_path = os.path.join(normalized_label_folder, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))

    # 이미지 크기 가져오기
    image = cv2.imread(image_path)
    height, width, _ = image.shape  # 이미지 높이, 너비
    
    # 정규화된 라벨 파일 읽기
    if os.path.exists(label_path):
        with open(label_path, 'r') as file:
            lines = file.readlines()
        
        absolute_lines = []
        
        for line in lines:
            values = line.strip().split()
            class_id = values[0]  # 클래스 ID
            coords = list(map(float, values[1:]))  # 정규화된 좌표
            
            # 정규화된 좌표 → 절대 좌표 변환
            absolute_coords = []
            for i in range(0, len(coords), 2):
                x_abs = int(coords[i] * width)  # x 좌표 복원
                y_abs = int(coords[i + 1] * height)  # y 좌표 복원
                absolute_coords.extend([x_abs, y_abs])
            
            # 변환된 데이터 저장
            absolute_line = f"{class_id} " + " ".join(map(str, absolute_coords))
            absolute_lines.append(absolute_line)
        
        # 변환된 좌표를 새 파일에 저장
        absolute_label_path = os.path.join(absolute_label_folder, os.path.basename(label_path))
        with open(absolute_label_path, 'w') as file:
            file.write("\n".join(absolute_lines))
