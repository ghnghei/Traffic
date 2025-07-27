# detection/yolov5_detector.py

from detection import detector_config as cfg
from ultralytics import YOLO
import cv2
import os


class YOLOv5Detector:
    def __init__(self, model_path=None,conf_thres=0.25):
        """
        初始化 YOLOv5 模型，默认使用配置文件中指定的模型路径
        """
        self.model_path = model_path if model_path else cfg.MODEL_PATH
        self.model = YOLO(self.model_path)
        self.conf_thres = conf_thres
    def detect_image(self, image_path, save_result=None, show_result=None):
        """
        检测单张图片（支持 PNG、JPG 等），返回预测结果对象

        参数:
            image_path (str): 图片路径
            save_result (bool): 是否保存结果，默认用配置文件中的设置
            show_result (bool): 是否显示结果窗口，默认用配置文件中的设置
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        save_result = cfg.SAVE_RESULT_IMAGE if save_result is None else save_result
        show_result = cfg.SHOW_RESULT_WINDOW if show_result is None else show_result

        # 传入置信度阈值参数conf
        results = self.model(image_path, conf=cfg.CONFIDENCE_THRESHOLD)

        if save_result:
            # 这里save会默认保存到runs/detect/predict/下，可以修改成保存到cfg.SAVE_DIRECTORY
            # 但ultralytics库目前默认路径不可直接改，需要后续处理
            results.save()

        if show_result:
            results.show()

        return results

    def detect_frame(self, frame, save_result=False, show_result=False):
        """
        检测一帧图像（如来自摄像头），返回预测结果

        参数:
            frame (numpy.ndarray): 单帧图像
            save_result (bool): 是否保存检测结果图像
            show_result (bool): 是否显示检测结果窗口
        """
        results = self.model.predict(source=frame, conf=cfg.CONFIDENCE_THRESHOLD, save=save_result, stream=False)

        if show_result:
            results.show()

        return results

    def detect_directory(self, folder_path, save_result=None, show_result=None):
        """
        批量检测一个文件夹下的所有图片

        参数:
            folder_path (str): 文件夹路径
            save_result (bool): 是否保存结果，默认用配置文件
            show_result (bool): 是否显示结果窗口，默认用配置文件
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")

        save_result = cfg.SAVE_RESULT_IMAGE if save_result is None else save_result
        show_result = cfg.SHOW_RESULT_WINDOW if show_result is None else show_result

        results = self.model(folder_path, conf=cfg.CONFIDENCE_THRESHOLD)

        if save_result:
            results.save()
        if show_result:
            results.show()

        return results
