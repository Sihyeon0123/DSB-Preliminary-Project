import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
# 디텍트론 2
import detectron2
# from detectron2.utils.logger import setup_logger
# setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
# yolo
from ultralytics import YOLO
# django
from django.apps import AppConfig
from PIL import Image, ImageDraw, ImageFont

class MyAppConfig(AppConfig):
    name = 'Server'

    def ready(self):
        if os.environ.get('RUN_MAIN', None) != 'true':
            return
        
        # YOLO모델 로드
        self.yolo_model = YOLO(r"C:\Users\705-18\Documents\GitHub\DSB-Preliminary-Project\Server\weights\안전장비 가중치.pt")
        print('YOLO모델이 성공적으로 로드되었습니다.')
        
        # # Detectron2 모델 로드
        # self.cfg = get_cfg()
        # self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        # self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  
        # self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        # self.cfg.MODEL.DEVICE = 'cuda'  # GPU 사용, CPU를 사용하려면 'cpu'
        
        # # 모델을 self에 저장
        # self.predictor = DefaultPredictor(self.cfg)
        # print("Detectron2 모델이 성공적으로 로드되었습니다.")

    def process_image_to_yolo(self, image, save_path):
        results = self.yolo_model.predict(image)

        # OpenCV 이미지를 Pillow 이미지로 변환
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)

        # 한글 폰트 로드 (Windows 기준, 경로 수정 필요)
        font = ImageFont.truetype("malgun.ttf", 20)

        # 결과를 기반으로 바운딩 박스 그리기    
        for result in results:
            boxes = result.boxes.xyxy  # 바운딩 박스 좌표
            class_ids = result.boxes.cls  # 클래스 ID

            # 이미지에 바운딩 박스 그리기
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                if int(class_ids[i]) == 0:
                    cls = '안전모'
                elif int(class_ids[i]) == 1:
                    cls = '안전대'
                else:
                    cls = '미착용'
                    
                # 바운딩 박스 그리기
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

                # 한글 텍스트 그리기
                label = f"ID: {cls}"
                draw.text((x1, y1 - 20), label, font=font, fill=(0, 0, 255))

         # Pillow 이미지를 다시 OpenCV 이미지로 변환
        processed_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # 결과 이미지 저장
        cv2.imwrite(save_path, processed_image)

        
    def process_image(self, image):
        # 이미지 예측
        outputs = self.predictor(image)
        
        # 결과 시각화
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # 결과 이미지를 반환 (RGB에서 BGR로 변환)
        return out.get_image()[:, :, ::-1]