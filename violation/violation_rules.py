import numpy as np


class ViolationRules:
    @staticmethod
    def check_red_light_violation(vehicle, traffic_light_status):
        """
        检查闯红灯违章
        :param vehicle: 车辆信息字典
        :param traffic_light_status: 红绿灯状态 ('red', 'green', 'yellow')
        :return: 是否违章
        """
        # 红灯状态且车辆越过停止线
        if traffic_light_status == 'red':
            # 假设车辆字典中有'stop_line_x'字段表示停止线x坐标
            if vehicle['position'][0] > vehicle.get('stop_line_x', 0) and vehicle['speed'] > 2:
                return True
        return False

    @staticmethod
    def check_reverse_driving(vehicle, lane_direction):
        """
        检查逆行违章
        :param vehicle: 车辆信息字典
        :param lane_direction: 车道方向向量 (numpy数组)
        :return: 是否违章
        """
        vehicle_dir = vehicle['direction_vector']
        if vehicle_dir is None or len(vehicle_dir) == 0:
            return False

        # 归一化向量
        vehicle_dir_norm = vehicle_dir / np.linalg.norm(vehicle_dir)
        lane_dir_norm = lane_direction / np.linalg.norm(lane_direction)

        # 计算车辆方向与车道方向的夹角余弦值
        dot_product = np.dot(vehicle_dir_norm, lane_dir_norm)

        # 如果夹角大于90度(余弦值小于0)则为逆行
        if dot_product < -0.3:  # 使用阈值避免误判
            return True
        return False

        if np.linalg.norm(vehicle_dir) == 0 or np.linalg.norm(lane_direction) == 0:
            return False

    @staticmethod
    def check_speeding(vehicle, speed_limit):
        """
        检查超速违章
        :param vehicle: 车辆信息字典
        :param speed_limit: 限速值(km/h)
        :return: 是否超速
        """
        # 将像素速度转换为实际速度（需要标定参数）
        calibrated_speed = vehicle['speed'] * 3.6  # 示例换算系数
        return calibrated_speed > speed_limit

    @staticmethod
    def check_illegal_lane_change(vehicle, lane_markings):
        """
        检查违规变道（压线行驶）
        :param vehicle: 车辆信息字典
        :param lane_markings: 车道线信息
        :return: 是否违章
        """
        # 简化实现：检查车辆是否跨越车道线
        if 'lane_id' in vehicle and 'prev_lane_id' in vehicle:
            if vehicle['lane_id'] != vehicle['prev_lane_id']:
                # 检查变道位置是否允许变道（如虚线处）
                # 实际应用中需要更复杂的逻辑
                return True
        return False