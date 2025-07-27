# detection/visualizer.py
import cv2
import numpy as np

def draw_detections(frame: np.ndarray, results) -> np.ndarray:
    """
    适配 Ultralytics YOLOv8 predict 返回的 results
    在图像上绘制检测框和标签
    """
    annotated_frame = frame.copy()

    if results is None or len(results) == 0:
        return annotated_frame

    # 取第一帧的检测结果
    res = results[0]

    boxes = res.boxes
    names = res.names

    if boxes is None or len(boxes) == 0:
        return annotated_frame

    # 遍历所有检测框
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
        conf = boxes.conf[i].cpu().item()
        cls_id = int(boxes.cls[i].cpu().item())
        label = f"{names[cls_id]} {conf:.2f}"

        # 画框
        cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        # 画标签
        cv2.putText(annotated_frame, label, (xyxy[0], xyxy[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return annotated_frame



# 车辆类别对应颜色，按需补充和调整
CLASS_COLOR_MAP = {
    'person': (0, 255, 0),        # 绿色
    'car': (255, 0, 0),           # 蓝色
    'bus': (0, 0, 255),           # 红色
    'truck': (255, 255, 0),       # 黄色
    'bicycle': (255, 0, 255),     # 紫色
    'motorcycle': (0, 255, 255),  # 青色
    'knife': (0, 128, 255),       # 橙色
    'fire': (0, 0, 128),          # 深红
    'explosive': (128, 0, 128),   # 紫红
}

DEFAULT_COLOR = (255, 255, 255)  # 白色框和文字备用色


def draw_boxes(image, results, class_color_map=None, conf_threshold=0.25):
    """
    在图像上绘制检测框与标签

    参数:
        image (np.ndarray): BGR图像
        results: YOLO检测结果对象，包含boxes、names、probs等
        class_color_map (dict): 类别对应颜色字典
        conf_threshold (float): 置信度阈值，低于则不绘制

    返回:
        img_out (np.ndarray): 绘制后的图像
    """
    if class_color_map is None:
        class_color_map = CLASS_COLOR_MAP

    img_out = image.copy()

    # 解析结果数据（Ultralytics YOLO接口不同版本可能不同，这里按常见格式处理）
    # results.boxes: BoundingBox对象列表，包含xyxy坐标和置信度
    # results.names: id->name映射
    # results.boxes.conf: 置信度列表
    # results.boxes.cls: 类别id列表

    if not hasattr(results, 'boxes') or len(results.boxes) == 0:
        return img_out  # 无检测框返回原图

    boxes = results.boxes
    names = results.names if hasattr(results, 'names') else {}

    for i, box in enumerate(boxes):
        conf = float(box.conf) if hasattr(box, 'conf') else 1.0
        cls_id = int(box.cls) if hasattr(box, 'cls') else -1
        cls_name = names.get(cls_id, str(cls_id))

        if conf < conf_threshold:
            continue

        # 框坐标，float转int
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # xyxy格式

        color = class_color_map.get(cls_name, DEFAULT_COLOR)

        # 绘制矩形框
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, thickness=2)

        # 标签文本，格式：“类别名 置信度”
        label = f"{cls_name} {conf:.2f}"

        # 计算文本大小和绘制背景
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_out, (x1, y1 - h - 6), (x1 + w, y1), color, -1)

        # 写文本
        cv2.putText(img_out, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return img_out
