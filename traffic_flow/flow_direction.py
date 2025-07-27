#方向检测模块
import numpy as np
from collections import defaultdict


class DirectionDetector:
    def __init__(self, line_points):
        """
        line_points: 计数线坐标 (x1, y1, x2, y2)
        """
        self.line_points = np.array(line_points)
        self.obj_history = defaultdict(list)
        self.direction_counts = defaultdict(lambda: defaultdict(int))

    @staticmethod
    def line_intersection(line1, line2):
        """计算两条线段的交点"""
        # 简化实现：计算向量叉积判断方向
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if det == 0:
            return None

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det

        return px, py

    def detect_direction(self, traffic_counter, track_id):
        """通过流量计数器获取轨迹历史并判断方向"""
        history = traffic_counter.get_track_history(track_id)

        if len(history) < 5:
            return "insufficient_data"

        # 获取最近两个位置点
        current_pos = np.array(history[-1]['position'])
        prev_pos = np.array(history[-2]['position'])

        # 计算移动向量
        movement = current_pos - prev_pos

        # 简化方向判断 (实际应使用计数线交点)
        if movement[1] > 1:  # Y坐标增加
            return "south"
        elif movement[1] < -1:  # Y坐标减少
            return "north"
        else:
            return "stationary"
                    