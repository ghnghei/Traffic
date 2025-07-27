import requests


class AlarmNotifier:
    def __init__(self, api_endpoint=None):
        """
        初始化警报通知器
        :param api_endpoint: 通知API端点
        """
        self.api_endpoint = api_endpoint

    def send_violation_alert(self, violation_data):
        """
        发送违章警报
        :param violation_data: 违章数据字典
        """
        # 控制台输出警报
        print(f"🚨 违章警报! 类型: {violation_data['violation_type']}")
        print(f"车辆ID: {violation_data['vehicle_id']} | 时间: {violation_data['timestamp']}")
        print(f"位置: {violation_data['location']} | 速度: {violation_data.get('speed', 0):.2f} km/h")
        print(f"截图路径: {violation_data['image_path']}")

        # 如果配置了API端点，则发送网络通知
        if self.api_endpoint:
            try:
                response = requests.post(
                    self.api_endpoint,
                    json=violation_data,
                    timeout=3
                )
                if response.status_code == 200:
                    print("通知发送成功")
                else:
                    print(f"通知发送失败，状态码: {response.status_code}")
            except Exception as e:
                print(f"通知发送异常: {str(e)}")

    def send_system_alert(self, message, level="warning"):
        """
        发送系统警报（如摄像头故障）
        :param message: 警报消息
        :param level: 警报级别 (info, warning, error)
        """
        print(f"[{level.upper()}] 系统警报: {message}")
        # 实际应用中可扩展为邮件/短信通知