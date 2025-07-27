#trajectory_analyzer.py
import numpy as np
import time


class TrajectoryAnalyzer:
    def __init__(self, max_history=15):
        """
        初始化轨迹分析器
        :param max_history: 最大历史记录帧数
        """
        self.vehicle_history = {}
        self.max_history = max_history
        self.trajectories = {}

    def update_trajectories(self, frame_id, vehicle_detections):
        """
        更新车辆轨迹历史
        :param frame_id: 当前帧ID
        :param vehicle_detections: 车辆检测结果列表
        """
        current_time = time.time()
        for vehicle in vehicle_detections:
            vid = vehicle['id']
            if vid not in self.vehicle_history:
                self.vehicle_history[vid] = []

            # 添加当前帧的位置和时间戳
            entry = {
                'frame_id': frame_id,
                'timestamp': current_time,
                'bbox': vehicle['bbox'],  # [x1, y1, x2, y2]
                'position': vehicle['position']  # (x, y)
            }

            self.vehicle_history[vid].append(entry)

            # 保持历史记录长度
            if len(self.vehicle_history[vid]) > self.max_history:
                self.vehicle_history[vid].pop(0)

    def calculate_speed(self, vehicle_id: int) -> float:

        """
        计算车辆速度 (像素/秒)
        :param vehicle_id: 车辆ID
        :return: 速度值
        """
        history = self.vehicle_history.get(vehicle_id, [])
        if len(history) < 2:
            return 0.0

        # 计算最近两个位置的距离和时间差
        pos1 = np.array(history[-1]['position'])
        pos2 = np.array(history[-2]['position'])
        distance = np.linalg.norm(pos1 - pos2)
        time_diff = history[-1]['timestamp'] - history[-2]['timestamp']

        if time_diff > 0:
            pixel_to_meter = 0.05  # 自行根据标定调整
            distance_meters = distance * pixel_to_meter
            return distance_meters / time_diff  # 单位: m/s
        return 0.0

    def get_direction_vector(self, vehicle_id):
        """
        获取车辆方向向量
        :param vehicle_id: 车辆ID
        :return: 方向向量
        """
        history = self.vehicle_history.get(vehicle_id, [])
        if len(history) < 2:
            return np.array([0.0, 0.0])


        # 计算最近两个位置的方向向量
        pos1 = np.array(history[-1]['position'])
        pos2 = np.array(history[-2]['position'])
        direction = pos1 - pos2

        if np.linalg.norm(direction) == 0:
            return np.array([0.0, 0.0])

        return direction
    def get_trajectory(self, vehicle_id):
        """
        获取车辆历史轨迹
        :param vehicle_id: 车辆ID
        :return: 轨迹点列表
        """
        history = self.vehicle_history.get(vehicle_id, [])
        return [entry['position'] for entry in history]

    def remove_inactive_trajectories(self, active_ids):
        # 删除轨迹字典里不在 active_ids 中的项，避免旧ID轨迹堆积
        inactive_ids = set(self.trajectories.keys()) - active_ids
        for track_id in inactive_ids:
            del self.trajectories[track_id]
