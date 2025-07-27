import cv2
import numpy as np
from detection.yolov5_detector import YOLOv5Detector

def get_brightness(frame):
    """计算图像亮度（V通道均值）"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2].mean()
    return brightness

def adjust_gamma(image, gamma=1.0):
    """Gamma 校正"""
    invGamma = 1.0 / gamma
    table = (255 * ((i / 255.0) ** invGamma) for i in range(256))
    table = np.array(list(table)).astype("uint8")
    return cv2.LUT(image, table)

def adjust_frame_by_light(frame, verbose=False):
    """
    根据图像亮度自动调整图像，适配光照条件
    :param frame: 原始图像
    :param verbose: 是否打印调试信息
    :return: 亮度调整后的图像
    """
    brightness = get_brightness(frame)
    if brightness < 80:
        if verbose: print("🌙 低光环境，增强图像亮度")
        frame = adjust_gamma(frame, gamma=1.5)
    elif brightness > 180:
        if verbose: print("☀️ 高光环境，降低图像亮度")
        frame = adjust_gamma(frame, gamma=0.7)
    else:
        if verbose: print("🌤 光照正常，无需调整")
    return frame

def realtime_detect_camera(camera_index=0, use_light_adjustment=True):
    """主函数：实时目标检测"""
    detector = YOLOv5Detector()
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    print("✅ 开始实时检测，按 'q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 无法读取摄像头帧")
            break

        # 动态光照适配
        if use_light_adjustment:
            frame = adjust_frame_by_light(frame, verbose=True)

        # 检测并绘制
        results = detector.detect_frame(frame)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv5 Realtime Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
