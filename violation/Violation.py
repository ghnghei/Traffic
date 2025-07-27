import cv2
import torch
import numpy as np
import time
from sort import Sort  # 需放置sort.py同目录
from violation.alarm_notifier import AlarmNotifier
from violation.trajectory_analyzer import TrajectoryAnalyzer
from violation.violation_rules import ViolationRules
from violation.violation_logger import ViolationLogger
from ultralytics import YOLO

model = YOLO("models/yolov5n.pt")
# 初始化
notifier = AlarmNotifier(api_endpoint=None)
tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)
trajectory_analyzer = TrajectoryAnalyzer(max_history=15)
violation_logger = ViolationLogger(log_dir="violation_logs")




# 摄像头打开
cap = cv2.VideoCapture(0)  # 0 表示默认摄像头

# 手动设置红绿灯状态（测试用）
traffic_light_status = "red"

# 假设一条车道合法方向向量示例 (向右)
lane_direction_vector = np.array([1, 0])
frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    # YOLOv5 预测，返回的 results.xyxy[0] 为 [x1, y1, x2, y2, conf, class] 格式张量
    # 推理结果
    results = model(frame)[0]  # results 是一个 Results 对象
    detections = []

    # 提取检测框
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls == 2:  # 只检测 car
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            detections.append([x1, y1, x2, y2, conf])

    # 转换为numpy数组给 SORT
    dets = np.array(detections)
    if dets.size == 0:
        dets = np.empty((0, 5))

    active_ids = set()

    # 跟踪更新
    tracked_objects = tracker.update(dets)
    vehicle_detections = []

    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj.astype(int)
        active_ids.add(track_id)

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        vehicle = {
            "id": int(track_id),
            "bbox": [x1, y1, x2, y2],
            "position": (center_x, center_y)
        }
        vehicle_detections.append(vehicle)
    trajectory_analyzer.remove_inactive_trajectories(active_ids)
    # 更新轨迹
    trajectory_analyzer.update_trajectories(frame_id, vehicle_detections)

    # 检查违规并绘制
    for vehicle in vehicle_detections:
        vehicle_id = vehicle["id"]
        speed = trajectory_analyzer.calculate_speed(vehicle_id)
        direction_vector = trajectory_analyzer.get_direction_vector(vehicle_id)
        vehicle["speed"] = speed
        vehicle["direction_vector"] = direction_vector

        violation_type = None
        if ViolationRules.check_red_light_violation(vehicle, traffic_light_status):
            violation_type = "Red Light"
        elif ViolationRules.check_reverse_driving(vehicle, lane_direction_vector):
            violation_type = "Reverse Driving"
        elif ViolationRules.check_speeding(vehicle, speed_limit=60):
            violation_type = "Speeding"

        x1, y1, x2, y2 = vehicle['bbox']

        if violation_type:
            log_entry = violation_logger.log_violation(frame, vehicle, violation_type, location="GPS:XXX")
            notifier.send_violation_alert(log_entry)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, violation_type, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            # ✅ 仅对当前活跃 ID 的车辆画绿色框
            if vehicle_id in active_ids:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Violation Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()