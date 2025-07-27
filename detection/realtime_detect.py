import cv2
from detection.yolov5_detector import YOLOv5Detector

def realtime_detect_camera(camera_index=0):
    detector = YOLOv5Detector()

    cap = cv2.VideoCapture(camera_index)  # 打开摄像头
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    print("✅ 开始实时检测，按 'q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 无法读取摄像头帧")
            break

        # 检测当前帧（结果为 Results 对象）
        results = detector.detect_frame(frame)

        # 可视化检测结果（在帧上绘制）
        annotated_frame = results[0].plot()  # 返回标注后的帧（np.ndarray）

        # 显示结果
        cv2.imshow("YOLOv5 Realtime Detection", annotated_frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
