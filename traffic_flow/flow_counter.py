# 流量计数核心（集成跟踪功能）
import cv2
import torch
from ultralytics import YOLO
from collections import defaultdict
import numpy as np



class TrafficCounter:
    def __init__(self, model_path='/home/orangepi/Desktop/traffic/models/yolov5n.pt', tracker='botsort.yaml'):
        """
        初始化带跟踪功能的流量计数器
        :param model_path: YOLOv5模型路径
        :param classes: 要检测的类别ID列表
        :param tracker: 跟踪器配置文件 (botsort.yaml/bytetrack.yaml)
        """
        torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
        self.model = YOLO(model_path)
        self.model.add_callback("on_predict_start", lambda x: setattr(x, "tracker", tracker))

        self.count_history = defaultdict(lambda: defaultdict(int))
        self.tracker_name = tracker
        self.class_map = {
            2: 'car',
            5: 'bus',
            7: 'truck',
            3: 'motorcycle',
            0: 'person'
        }
        self.classes = [2, 5, 7]
        self.track_history = defaultdict(list)  # 存储跟踪对象历史轨迹

    def process_frame(self, frame):
        """处理视频帧并返回带跟踪ID的检测结果"""
        #使用跟踪模式 (关键修改点)
        #执行跟踪检测
        results = self.model.track(frame, persist=True, verbose=False)

        #解析结果
        detections = []
        current_counts = defaultdict(int)
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else None

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    conf = confidences[i]
                    cls_id = class_ids[i]
                    track_id = track_ids[i] if track_ids is not None else i

                    # 只处理目标类别
                    if cls_id in self.classes:
                        vehicle_type = self.class_map.get(cls_id, f"class_{cls_id}")
                        current_counts[vehicle_type] += 1

                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'class_id': int(cls_id),
                            'track_id': int(track_id),
                            'type': vehicle_type
                        })

        for det in detections:
            if len(det) < 7:  # 确保有跟踪ID字段
                continue

            cls_id = int(det[5])
            track_id = int(det[6])

            if cls_id in self.classes:
                vehicle_type = self.class_map.get(cls_id, f"class_{cls_id}")
                current_counts[vehicle_type] += 1

                # 存储带跟踪ID的检测对象
                tracked_objects.append({
                    'bbox': det[:6],  # [x1, y1, x2, y2, conf, cls_id]
                    'track_id': track_id,
                    'type': vehicle_type
                })

                # 更新轨迹历史 (用于方向检测)
                self.track_history[track_id].append({
                    'frame_time': datetime.now(),
                    'position': ((det[0] + det[2]) / 2, det[3])  # (中心X, 底部Y)
                })

        return {
            'frame': results[0].plot(),  # 渲染后的帧
            'counts': dict(current_counts),
            'detections': detections
        }

    def update_history(self, counts, timestamp):
        """更新历史计数数据 (保持不变)"""
        for vehicle_type, count in counts.items():
            self.count_history[timestamp][vehicle_type] += count

    def get_total_count(self):
        """获取总计数 (保持不变)"""
        total = defaultdict(int)
        for time_data in self.count_history.values():
            for vehicle_type, count in time_data.items():
                total[vehicle_type] += count
        return dict(total)

    def get_track_history(self, track_id):
        """获取特定跟踪ID的历史轨迹"""
        return self.track_history.get(track_id, [])