import detectron2
# from detectron2.utils.logger import setup_logger
# setup_logger()
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import copy
import requests
import io

import warnings
# 특정 UserWarning을 무시
warnings.filterwarnings("ignore", category=UserWarning)


def calculate_angle(point1, point2, point3):
    """
        각도 계산 함수
    """
    # 방향 벡터 계산
    vector1 = np.array(point2) - np.array(point1)  # p1 -> p2
    vector2 = np.array(point3) - np.array(point2)  # p2 -> p3
    
    # 내적 계산
    dot_product = np.dot(vector1, vector2)
    
    # 크기 계산
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # 코사인 각도 계산
    cos_angle = dot_product / (magnitude1 * magnitude2)
    
    # 각도 (라디안) 계산
    angle_rad = np.arccos(cos_angle)
    
    # 라디안에서 도로 변환
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def min_max_scale(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

skeleton_connections = [
            (0, 1), (0, 2),        # 코에서 두 눈으로 연결
            (1, 3), (2, 4),        # 왼쪽/오른쪽 눈에서 귀로 연결
            (5, 6),                # 왼쪽 어깨와 오른쪽 어깨 연결
            (5, 7), (7, 9),        # 왼쪽 어깨에서 팔꿈치, 팔꿈치에서 손목으로 연결
            (6, 8), (8, 10),       # 오른쪽 어깨에서 팔꿈치, 팔꿈치에서 손목으로 연결
            (5, 11), (6, 12),      # 왼쪽 어깨에서 왼쪽 엉덩이, 오른쪽 어깨에서 오른쪽 엉덩이로 연결
            (11, 12),              # 왼쪽 엉덩이와 오른쪽 엉덩이 연결
            (11, 13), (13, 15),    # 왼쪽 엉덩이에서 무릎, 무릎에서 발목 연결
            (12, 14), (14, 16)     # 오른쪽 엉덩이에서 무릎, 무릎에서 발목 연결
        ]


if __name__ == '__main__':
    URL = "http://localhost:8000/call/"

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE='cuda' #만약 gpu를 사용한다면 ‘cuda’로 수정
    predictor = DefaultPredictor(cfg)
    print('Detectron2 모델 로드 성공')

    # df = pd.read_excel('t.xlsx')
    df = pd.read_excel(r'C:\Users\USER\Documents\GitHub\DSB-Preliminary-Project\SafeWatch\t.xlsx')

    X = df.drop(columns=['label'])  # 특징 (feature)
    y = df['label']   

    # 훈련 세트와 테스트 세트로 나누기
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print('분류 모델 로드 성공')

    # 이미지들이 저장된 폴더 경로
    image_folder = 'drop/'

    # 이미지 파일 목록을 불러오기
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')])

    temp_que = []
    i = 0
    # 이미지 처리 루프
    for image_file in image_files:
        # 이미지 경로 생성
        image_path = os.path.join(image_folder, image_file)
        # 이미지 읽어오기
        frame = cv2.imread(image_path)
        # 이미지 크기 가져오기
        image_height, image_width, channels = frame.shape
        
        if frame is None:
            print(f"이미지 로드에 실패했습니다: {image_file}")
            continue
        
        # Detectron2를 이용한 관절 검출
        # keypoints = predictor(frame)['instances'].get_fields()['pred_keypoints'][0]

        outputs = predictor(frame)
        fields = outputs['instances'].get_fields()

        keypoints = fields['pred_keypoints'][0]
        bounding_boxes = fields['pred_boxes'][0]  # 바운딩 박스 좌표

        # 관절 각도 계산
        # 팔꿈치 각도
        elbow_angle = calculate_angle((int(keypoints[6][0]), int(keypoints[6][1])), (int(keypoints[8][0]), int(keypoints[8][1])), (int(keypoints[10][0]), int(keypoints[10][1])))  
        # 어깨 각도 
        # 어깨 각도를 오른 팔꿈치-오른 어깨-왼쪽 어깨 에서 오른 팔꿈치-오른 어깨-오른 엉덩이로 바꿈
        shoulder_angle = calculate_angle((int(keypoints[6][0]), int(keypoints[6][1])), (int(keypoints[8][0]), int(keypoints[8][1])), (int(keypoints[12][0]), int(keypoints[12][1]))) 
        # 허리 각도 
        hip_angle = calculate_angle((int(keypoints[6][0]), int(keypoints[6][1])), (int(keypoints[12][0]), int(keypoints[12][1])), (int(keypoints[13][0]), int(keypoints[13][1])))  
        # 무릎 각도
        knee_angle = calculate_angle((int(keypoints[12][0]), int(keypoints[12][1])), (int(keypoints[14][0]), int(keypoints[14][1])), (int(keypoints[16][0]), int(keypoints[16][1])))  

        # 높이(y) 추출
        head_y = min_max_scale(int(keypoints[0][1]), 0, image_height)
        hip_y = min_max_scale(int(keypoints[12][1]), 0, image_height)
        ankle_y = min_max_scale(int(keypoints[16][1]), 0, image_height)
        
        f_lst = [elbow_angle, shoulder_angle, hip_angle, knee_angle, head_y, hip_y, ankle_y]


        # 프레임 카운트
        if i < 5:
            i += 1
            for f in f_lst:
                temp_que.append(f)
        elif i <= 5:
            temp_que = temp_que[7:]
            for f in f_lst:
                temp_que.append(f)

            temp_df = pd.DataFrame([temp_que], columns=['elbow_angle1', 'shoulder_angle1', 'hip_angle1', 'knee_angle1', 'head_y1', 'hip_y1', 'ankle_y1',
                                                        'elbow_angle2', 'shoulder_angle2', 'hip_angle2', 'knee_angle2', 'head_y2', 'hip_y2', 'ankle_y2',
                                                        'elbow_angle3', 'shoulder_angle3', 'hip_angle3', 'knee_angle3', 'head_y3', 'hip_y3', 'ankle_y3',
                                                        'elbow_angle4', 'shoulder_angle4', 'hip_angle4', 'knee_angle4', 'head_y4', 'hip_y4', 'ankle_y4',
                                                        'elbow_angle5', 'shoulder_angle5', 'hip_angle5', 'knee_angle5', 'head_y5', 'hip_y5', 'ankle_y5'])
            
            test_pred = model.predict(temp_df)
            result= test_pred[0]

            # if result == 0:
            #     print('정상')
            #     data = {'value': 0}
            data = None
            if result == 1:
                print('쓰러짐')
                data = {'value': 1}
            elif result == 2:
                print('떨어짐')
                data = {'value': 2}

            if data is not None:
                # 바운딩 박스 좌표 가져오기
                bbox = bounding_boxes.tensor.cpu().numpy().astype(int)[0]
                # 바운딩 박스 그리기
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # 이미지를 메모리에 저장
                _, buffer = cv2.imencode('.jpg', frame)
                image_file = io.BytesIO(buffer)

                # 이미지 파일을 서버에 전송
                files = {'file': ('output_image_with_bbox.jpg', image_file, 'image/jpeg')}
                response = requests.post(URL, data=data,  files=files)
                if response.status_code == 200:
                    print('정상 전송')
                else:
                    print('실패')
                break
