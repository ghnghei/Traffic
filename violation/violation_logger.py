import os
import json
import cv2
from datetime import datetime


class ViolationLogger:
    def __init__(self, log_dir="violation_logs"):
        """
        初始化违章记录器
        :param log_dir: 日志存储目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def log_violation(self, frame, vehicle, violation_type, location):
        """
        记录违章信息并保存截图
        :param frame: 视频帧
        :param vehicle: 车辆信息
        :param violation_type: 违章类型
        :param location: 地理位置(GPS坐标)
        :return: 违章日志字典
        """
        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{violation_type}_{vehicle['id']}_{timestamp}"

        # 保存违章截图
        img_path = os.path.join(self.log_dir, f"{filename}.jpg")
        cv2.imwrite(img_path, self._annotate_frame(frame, vehicle, violation_type))

        # 创建日志条目
        log_entry = {
            "timestamp": timestamp,
            "vehicle_id": vehicle['id'],
            "violation_type": violation_type,
            "location": location,
            "speed": vehicle.get('speed', 0),
            "image_path": img_path,
            "direction": vehicle.get('direction_vector', []).tolist() if 'direction_vector' in vehicle else []
        }

        # 保存到JSON日志
        log_path = os.path.join(self.log_dir, f"{filename}.json")
        with open(log_path, 'w') as f:
            json.dump(log_entry, f, indent=2)

        return log_entry

    def _annotate_frame(self, frame, vehicle, violation_type):
        """
        在帧上标注违章信息
        :param frame: 原始视频帧
        :param vehicle: 车辆信息
        :param violation_type: 违章类型
        :return: 标注后的帧
        """
        annotated = frame.copy()
        x1, y1, x2, y2 = vehicle['bbox']

        # 绘制边界框
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 添加违章信息文本
        text = f"{violation_type} - ID:{vehicle['id']}"
        cv2.putText(annotated, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 绘制方向箭头
        if 'direction_vector' in vehicle and vehicle['direction_vector'] is not None:
            direction = vehicle['direction_vector']
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            end_point = (int(center[0] + direction[0] * 20), int(center[1] + direction[1] * 20))
            cv2.arrowedLine(annotated, center, end_point, (0, 255, 255), 2)

        return annotated