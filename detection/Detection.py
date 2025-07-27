import cv2
import time
import numpy as np
import threading
from detection.detector_config import load_detector_config
from detection.yolov5_detector import YOLOv5Detector
from detection.visualizer import draw_detections

# 全局变量
last_results = None
frame_queue = []
processing_frame = None
frame_lock = threading.Lock()
running = True
real_fps = 0.0
fps_counter = 0
fps_timer = time.time()
target_size = 160  # 更小的目标尺寸
skip_frames = 3  # 跳过的帧数


def detection_worker(detector, width, height):
    global last_results, frame_queue, processing_frame, running

    print("🔧 启动检测工作线程...")
    while running:
        with frame_lock:
            if not frame_queue:
                time.sleep(0.01)  # 短暂休眠减少CPU占用
                continue

            # 获取队列中的最新帧
            processing_frame = frame_queue[-1]
            # 清空队列只保留最新帧
            frame_queue = []

        # 缩小图像尺寸
        small_frame = cv2.resize(processing_frame, (target_size, target_size))

        try:
            # 执行检测
            results = detector.detect_frame(small_frame)

            # 处理结果
            processed_results = []
            if hasattr(results, 'boxes') and results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    conf = confidences[i]
                    cls_id = class_ids[i]

                    # 缩放回原始尺寸
                    scale_x = width / target_size
                    scale_y = height / target_size
                    x1, x2 = x1 * scale_x, x2 * scale_x
                    y1, y2 = y1 * scale_y, y2 * scale_y

                    processed_results.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class_id': int(cls_id),
                        'label': results.names[cls_id]
                    })

            # 只保留置信度最高的目标
            if processed_results:
                processed_results.sort(key=lambda x: x['confidence'], reverse=True)
                last_results = [processed_results[0]]
            else:
                last_results = None

        except Exception as e:
            print(f"检测错误: {e}")
            last_results = None


def main(camera_index=0):
    global last_results, frame_queue, running, real_fps, fps_counter, fps_timer

    print("⚡️ 车辆检测系统 - 性能优化版")
    config = load_detector_config()

    # 创建检测器
    detector = YOLOv5Detector(
        model_path=config['model_path'],
        conf_thres=config['conf_thres']
    )

    # 设置摄像头
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 获取实际分辨率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头分辨率: {width}x{height}")

    # 设置更低的FPS上限（节省CPU）
    cap.set(cv2.CAP_PROP_FPS, 15)

    print(f"✅ 优化策略: 极小尺寸处理({target_size}px) | 多线程 | 动态跳帧(每{skip_frames + 1}帧处理1帧)")
    print("🎥 开始实时检测 (按 'q' 退出)")

    # 创建窗口
    cv2.namedWindow("车辆检测", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("车辆检测", width, height)

    # 预热模型
    print("预热模型中...")
    dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
    detector.detect_frame(dummy_frame)
    print("✅ 模型预热完成")

    # 启动检测线程
    detection_thread = threading.Thread(
        target=detection_worker,
        args=(detector, width, height),
        daemon=True
    )
    detection_thread.start()

    frame_count = 0

    try:
        while True:
            # FPS计算
            current_time = time.time()
            fps_counter += 1
            if current_time - fps_timer >= 1.0:
                real_fps = fps_counter / (current_time - fps_timer)
                fps_counter = 0
                fps_timer = current_time

            # 读取帧
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # 水平翻转（可选）
            # frame = cv2.flip(frame, 1)

            # 优化: 跳过部分帧
            frame_count += 1
            if frame_count % skip_frames != 0:
                # 不处理此帧，直接显示
                display_frame = frame.copy()
                if last_results:
                    display_frame = draw_detections(display_frame, last_results)

                # 显示FPS
                cv2.putText(display_frame, f"FPS: {real_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("车辆检测", display_frame)

                # 轻量级退出检查
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("👋 用户退出")
                    break
                continue

            # 将帧添加到队列
            with frame_lock:
                frame_queue.append(frame.copy())

            # 显示当前帧（使用上次结果）
            display_frame = frame.copy()
            if last_results:
                display_frame = draw_detections(display_frame, last_results)

            # 显示FPS
            cv2.putText(display_frame, f"FPS: {real_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("车辆检测", display_frame)

            # 轻量级退出检查
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 用户退出")
                break

    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        # 保存错误帧用于调试
        cv2.imwrite("error_frame.jpg", frame)
    finally:
        running = False
        detection_thread.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()