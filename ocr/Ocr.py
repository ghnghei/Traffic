import cv2
import time
import numpy as np
from ocr.ocr_engine import PlateRecognizer
from ocr.ocr_utils import preprocess_image, postprocess_text
from ocr.plate_cropper import PlateCropper


def draw_plate_info(frame, plate_text, bbox, confidence):
    """在图像上绘制识别出的车牌信息"""
    if not plate_text:
        return

    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    label = f"{plate_text} ({confidence:.2f})"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def main(camera_index=0):
    print("📸 车牌识别系统启动 (不使用检测模型)")

    # 初始化车牌识别器
    plate_recognizer = PlateRecognizer(
        use_gpu=False,
        lang='ch',
        model_dir='/home/orangepi/Desktop/traffic/models/Paddle'  # 模型路径
    )

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    print("🎥 开始实时车牌识别 (按 'q' 退出)")
    cv2.namedWindow("车牌识别", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 直接对整帧进行识别（或者你可以加一个简单裁剪区域）
        plate_text, bbox, conf = plate_recognizer.recognize_plate(frame, (0, 0, frame.shape[1], frame.shape[0]))

        # 显示识别结果
        if plate_text:
            draw_plate_info(frame, plate_text, bbox, conf)
            print(f"📛 识别结果: {plate_text} ({conf:.2f})")

        cv2.imshow("车牌识别", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 退出程序")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
