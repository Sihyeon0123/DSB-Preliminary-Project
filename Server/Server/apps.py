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

from django.apps import AppConfig

class MyAppConfig(AppConfig):
    name = 'Server'

    def ready(self):
        if os.environ.get('RUN_MAIN', None) != 'true':
            return
        
        # Detectron2 모델 로드
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.DEVICE = 'cuda'  # GPU 사용, CPU를 사용하려면 'cpu'
        
        # 모델을 self에 저장
        self.predictor = DefaultPredictor(self.cfg)
        print("Detectron2 모델이 성공적으로 로드되었습니다.")

        
    def process_image(self, image):
        # 이미지 예측
        outputs = self.predictor(image)
        
        # 결과 시각화
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # 결과 이미지를 반환 (RGB에서 BGR로 변환)
        return out.get_image()[:, :, ::-1]