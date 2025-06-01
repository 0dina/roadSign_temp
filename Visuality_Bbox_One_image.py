import cv2
import numpy as np
'''
사각형 바운딩 박스와 폴리곤 바운딩 박스 두가지 형태의 바운딩 박스 모두 시각화 가능
'''
# 이미지와 텍스트 파일 경로 설정
image_path = '/Users/iseungju/Desktop/Making_Data/data_v1/valid/images/signdataset-11-_jpg.rf.992ea147eebe6612c467bd3084d24672.jpg'  # 이미지 경로
input_txt_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
# input_txt_path = 'test_data/labels/30km_limit.txt'

# 이미지 읽기
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# 이미지 크기 가져오기
height, width, _ = image.shape

# 텍스트 파일에서 바운딩 박스 좌표 읽기
with open(input_txt_path, "r") as file:
    lines = file.readlines()

# 바운딩 박스를 이미지에 시각화
for line in lines:
    # 각 좌표를 공백을 기준으로 분리하여 리스트로 변환
    coordinates = list(map(float, line.split()))  # 실수형으로 변환
    
    # 첫 번째 값은 레이블 (라벨 출력)
    label = int(coordinates[0])  # 레이블은 정수형
    print(f"Label: {label}")
    
    # 두 번째 값부터는 좌표들
    coords = coordinates[1:]
    
    # **사각형 바운딩 박스 처리**
    if len(coords) == 4:  # x, y, width, height
        x_center = coords[0]
        y_center = coords[1]
        box_width = coords[2]
        box_height = coords[3]
        
        # 정규화된 경우 이미지 크기 기준으로 복원
        if all(0 <= val <= 1 for val in coords):
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height
        
        # 좌상단 좌표 계산
        x_min = int(x_center - box_width / 2)
        y_min = int(y_center - box_height / 2)
        x_max = int(x_center + box_width / 2)
        y_max = int(y_center + box_height / 2)
        
        # 사각형 그리기
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)
        cv2.putText(image, f"Label: {label}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # **폴리곤 바운딩 박스 처리**
    elif len(coords) >= 6 and len(coords) % 2 == 0:  # 최소 3개 점 필요 (x1, y1, x2, y2, x3, y3)
        absolute_coords = []
        
        # 정규화된 좌표 처리
        if all(0 <= val <= 1 for val in coords):
            for i in range(0, len(coords), 2):
                x = int(coords[i] * width)
                y = int(coords[i + 1] * height)
                absolute_coords.append((x, y))
        else:  # 정수 좌표 처리
            for i in range(0, len(coords), 2):
                x = int(coords[i])
                y = int(coords[i + 1])
                absolute_coords.append((x, y))
        
        # 폴리곤 그리기
        polygon = np.array(absolute_coords, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(image, f"Label: {label}", (absolute_coords[0][0], absolute_coords[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    else:
        print(f"Invalid format for label {label} in line: {line.strip()}")

# 결과 이미지 시각화
cv2.imshow("Bounding Boxes Visualization", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
