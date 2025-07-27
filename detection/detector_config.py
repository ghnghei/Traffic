# detection/detector_config.py

import os

# YOLOv5 模型路径（可使用 yolov5n.pt 或你训练好的自定义模型）
MODEL_PATH = 'yolov5n.pt'

# 检测图片路径（可在 main.py 中使用）
DEFAULT_IMAGE_PATH = os.path.join('images', 'test.png')

# 支持的图片格式（用于文件夹批量检测）
SUPPORTED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp')

# 是否显示检测窗口
SHOW_RESULT_WINDOW = True

# 是否保存检测结果图像
SAVE_RESULT_IMAGE = True

# 输出保存路径（YOLO 默认保存在 runs/detect）
SAVE_DIRECTORY = 'runs/detect/predict'

# 类别筛选（如仅检测这几类，空列表代表检测所有）
# 可选：'person', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'knife', 'fire', 'explosive'
FILTER_CLASSES = []

# 最小置信度阈值（只显示高于这个置信度的检测结果）
CONFIDENCE_THRESHOLD = 0.25
def load_detector_config():
    return {
        'model_path': "../models/yolov5n.pt",
        'conf_thres': 0.25
    }
