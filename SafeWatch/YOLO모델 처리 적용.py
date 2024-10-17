import numpy as np
import os, json, cv2, random
import pandas as pd
import numpy as np
import copy
import requests
import io
import threading
import warnings
# 특정 UserWarning을 무시
warnings.filterwarnings("ignore", category=UserWarning)

# YOLO 모델 
from ultralytics import YOLO

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

def send_data_to_server(data, frame):
    # 이미지를 메모리에 저장
    _, buffer = cv2.imencode('.jpg', frame)
    image_file = io.BytesIO(buffer)

    # 이미지 파일을 서버에 전송
    files = {'file': ('output_image_with_bbox.jpg', image_file, 'image/jpeg')}
    
    # 서버에 POST 요청 전송
    response = requests.post(URL, data=data, files=files)
    
    # 응답 처리
    if response.status_code == 200:
        print('정상 전송')
    else:
        print('실패:', response.text)

if __name__ == '__main__':
    URL = "http://localhost:8000/call/"
    # folder_path = './no_helmet'   
    folder_path = './f'  

    # # YOLO모델 로드
    safety_model = YOLO(r".\weights\안전장비 가중치.pt")
    fire_model = YOLO(r".\weights\화재감지 가중치.pt")
    print('YOLO모델이 성공적으로 로드되었습니다.')

    # 비디오가 있는 경로에서 이미지를 가져온다
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 이미지 파일이 있는지 확인
    if not image_files:
        print("이미지가 없습니다.")
        exit()

    
    # 불이나 연기가 몇번이나 감지되었는지 카운트
    fire_frame_cnt = 0
    # 화재 상태에서 연기가 몇번이나 감지 X인지 카운트
    no_fire_frame_cnt = 0
    # 화재상태임을 알림
    fire_detection_state = False
    # 화재 알림 유무
    is_fire_notified = False

    # 슬라이드쇼 시작
    for image_file in image_files:
        # 이미지 읽기
        image_path = os.path.join(folder_path, image_file)
        frame = cv2.imread(image_path)

        # 이미지가 정상적으로 읽혔는지 확인
        if frame is None:
            print(f"이미지를 불러올 수 없습니다: {image_path}")
            continue

        # 이미지 처리 로직
        # 화재 탐지
        fire_detection_results = fire_model.predict(image_path, verbose=False, conf=0.5)
        fire_boxes = fire_detection_results[0].boxes if len(fire_detection_results) > 0 else None
        # 탐지된 객체가 존재하는지 확인
        if fire_boxes and len(fire_boxes) > 0:
            # 탐지된 객체에 바운딩박스 그리기
            for box in fire_boxes:
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                cls = int(box.cls)
                # 바운딩 박스 그리기
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
                # 이름 그리기
                if cls == 0:
                    name = 'fire'
                else:
                    name = 'smoke'
                # 클래스 이름과 신뢰도 텍스트 그리기

                cv2.putText(frame, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            
            # 화재가 아닌 상태에서 다섯번 이상 감지었다면
            if fire_frame_cnt >= 5 and not fire_detection_state:
                print('화재가 감지되었습니다.')
                fire_detection_state = True
            else:
                # 화재가 감지되지 않은 상태라면
                if not fire_detection_state:
                    fire_frame_cnt += 1
            #  화재 객체가 감지되면 카운트 초기화
            no_fire_frame_cnt = 0

        # 화재 상태에서 벗어나기 위한 조건문
        if fire_detection_state:
            fire_frame_cnt = 0
            # 연속 5번 이상 화재가 감지되지 않으면 화재상태 해제
            if no_fire_frame_cnt >= 5:
                fire_detection_state = False
                is_fire_notified = True
                print('화재상태가 해제되었습니다.')
            else:
                no_fire_frame_cnt += 1


        # 안전장비 탐지
        # 안전장비가 연속으로 잡히지 않은 경우
        no_safety_gear_cnt = 0
        # 안전장비 미착용 작업자가 있는 상태인지 유무
        no_safety_gear_state = False
        # 안전장비 미착용 유무 알림 여부
        is_no_safety_gear_notified = False

        safety_gear_results = safety_model.predict(image_path, verbose=False, conf=0.5)
        safety_gear_boxes = safety_gear_results[0].boxes if len(safety_gear_results) > 0 else None
        # 탐지된 객체가 존재하는지 확인
        if safety_gear_boxes and len(safety_gear_boxes) > 0:
            # 탐지된 객체에 바운딩박스 그리기
            for box in safety_gear_boxes:
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                cls = int(box.cls)
                name = ''
                if cls == 0:
                    name = 'helmet'
                elif cls == 1:
                    name = 'belt'
                elif cls == 2:
                    name = 'no_helmet'
                elif cls == 3:
                    name = 'no_belt'

                # 바운딩 박스 그리기
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)
                cv2.putText(frame, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
                
            # 탐지된 객체 리스트 
            detect_safety_gear_lst = safety_gear_boxes[0].cls.tolist()
            # 안전모 미착용자가 있다면
            if 2 in detect_safety_gear_lst or 3 in detect_safety_gear_lst:
                no_safety_gear_cnt += 1
            # 안전모와 안전대 둘다 잡혔다면
            elif 0 in detect_safety_gear_lst and 1 in detect_safety_gear_lst\
                and 2 not in detect_safety_gear_lst and 3 not in detect_safety_gear_lst\
                and not no_safety_gear_state:
                no_safety_gear_cnt = 0
                no_safety_gear_state = False
                is_no_safety_gear_notified = False

            # 만약 5번 연속으로 미착용자가 잡혔다면
            if no_safety_gear_cnt >= 5 and not no_safety_gear_state:
                print('미착용자가 발견되었습니다.')
                no_safety_gear_state = True

        # 결과 서버 전송
        # 만약 안전장비 미착용자가 발견되고 알림을 전송하지 않았다면 or
        # 만약 화재상태이고 알림을 전송하지 않았다면
        # 0: 쓰러짐, 1: 추락, 2: 화재, 3: 안전장비 미착용
        if (no_safety_gear_state and not is_no_safety_gear_notified) or (fire_detection_state and not is_fire_notified):
            result_satate = []
            # 화재 발생
            if fire_detection_state:
                result_satate.append(2)    
            is_fire_notified = True

            # 안전장비 미착용
            if no_safety_gear_state:
                result_satate.append(3)
            is_no_safety_gear_notified = True

            # 결과 생성
            data = {'value': result_satate} 
            print(data)
            # 쓰레드 생성 및 시작
            thread = threading.Thread(target=send_data_to_server, args=(data, frame))
            thread.start()


        # 이미지 출력
        cv2.imshow('video', frame)

        # 빠르게 이미지를 교체 (1ms)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    # 창 종료
    cv2.destroyAllWindows()






