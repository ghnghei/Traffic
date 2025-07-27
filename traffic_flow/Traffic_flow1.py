import cv2
import time
import numpy as np
import threading
from collections import defaultdict
from datetime import datetime
from detection.visualizer import draw_detections
from traffic_flow.flow_direction import DirectionDetector
from traffic_flow.flow_counter import TrafficCounter  # 使用新的流量计数器

# 全局变量
running = True
real_fps = 0.0
fps_counter = 0
fps_timer = time.time()

# 交通统计相关全局变量
traffic_counter = TrafficCounter(
    model_path='../models/yolov5n.pt',
)
direction_detector = None
traffic_data_lock = threading.Lock()
traffic_data = {
    'total_counts': defaultdict(int),
    'direction_counts': defaultdict(lambda: defaultdict(int)),
    'count_history': defaultdict(lambda: defaultdict(int))
}


def generate_report():
    """生成并保存交通统计报告"""
    global traffic_data
    if not traffic_data['total_counts']:
        return "无交通数据可生成报告"

    # 创建报告生成器
    from report_generator import ReportGenerator
    report_gen = ReportGenerator(output_dir="traffic_reports")

    # 生成CSV报表
    csv_path = report_gen.generate_csv_report(traffic_data['count_history'])

    # 生成可视化报表
    count_path, dir_path = report_gen.generate_visual_report(
        traffic_data['count_history'],
        traffic_data['direction_counts']
    )

    # 生成文本摘要
    summary = report_gen.generate_summary(
        traffic_data['total_counts'],
        traffic_data['direction_counts']
    )

    # 保存文本摘要
    text_report_path = f"traffic_reports/traffic_summary_{int(time.time())}.txt"
    with open(text_report_path, 'w') as f:
        f.write(summary)

    return f"报告已生成:\nCSV: {csv_path}\n图表: {count_path}, {dir_path}\n摘要: {text_report_path}"


def main(camera_index=0):
    global running, real_fps, fps_counter, fps_timer
    global traffic_counter, direction_detector, traffic_data

    print("⚡⚡⚡⚡⚡⚡⚡⚡️ 车辆检测与流量统计系统")

    # 设置摄像头
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 获取实际分辨率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头分辨率: {width}x{height}")

    # 设置计数线（根据实际场景调整）
    line_points = [0, height // 2, width, height // 2]  # 水平中线
    print(f"计数线位置: ({line_points[0]}, {line_points[1]}) -> ({line_points[2]}, {line_points[3]})")

    # 初始化流量统计和方向检测模块
    print("⏳ 正在加载车辆检测模型和跟踪器...")
    traffic_counter = TrafficCounter(tracker='botsort.yaml') # 使用带跟踪的流量计数器
    print("✅ 车辆检测模型加载完成")
    direction_detector = DirectionDetector(line_points)
    print("✅ 方向检测器初始化完成")

    # 清空历史数据
    with traffic_data_lock:
        traffic_data = {
            'total_counts': defaultdict(int),
            'direction_counts': defaultdict(lambda: defaultdict(int)),
            'count_history': defaultdict(lambda: defaultdict(int))
        }

    # 设置FPS上限
    cap.set(cv2.CAP_PROP_FPS, 15)

    print(f"✅ 优化策略: 多线程 | 跟踪功能已启用")
    print("🎥🎥🎥🎥🎥🎥🎥🎥 开始实时检测与流量统计 (按 'q' 退出, 按 'r' 生成报告)")

    # 创建窗口
    cv2.namedWindow("车辆检测与流量统计", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("车辆检测与流量统计", width, height)

    # 预热模型
    print("预热模型中...")
    try:
        # 使用小型彩色图像而非全黑图像
        dummy_frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        traffic_counter.process_frame(dummy_frame)
        print("✅ 模型预热完成")
    except Exception as e:
        print(f"⚠️ 模型预热失败: {e}")
        print("⚠️ 尝试跳过预热继续运行...")
    print("✅ 模型预热完成")

    frame_count = 0
    last_report_time = time.time()
    auto_report_interval = 300  # 每5分钟自动生成报告

    first_frame_timeout = 5  # 5秒超时
    start_time = time.time()
    final_counts = None  # 初始化返回值

    try:
        while True:
            current_time = time.time()
            fps_counter += 1
            if current_time - fps_timer >= 1.0:
                real_fps = fps_counter / (current_time - fps_timer)
                fps_counter = 0
                fps_timer = current_time

            # 定期自动生成报告
            if current_time - last_report_time > auto_report_interval:
                report_info = generate_report()
                print(f"\n⏱⏱⏱⏱⏱⏱⏱⏱⏱ 定期报告生成: {report_info}")
                last_report_time = current_time
                # 重置统计数据
                with traffic_data_lock:
                    traffic_data['count_history'] = defaultdict(lambda: defaultdict(int))

            # 读取帧
            ret, frame = cap.read()
            if not ret:
                if time.time() - start_time > first_frame_timeout:
                    print(f"❌ 无法从摄像头获取帧，请检查摄像头连接（{camera_index}）")
                    break
                time.sleep(0.05)
                continue

            frame_count += 1

            # 处理帧并获取结果
            result = traffic_counter.process_frame(frame)

            # 更新历史计数
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            traffic_counter.update_history(result['counts'], timestamp)

            # 显示当前帧
            display_frame = result['frame'].copy()

            # 绘制计数线
            cv2.line(display_frame,
                     (line_points[0], line_points[1]),
                     (line_points[2], line_points[3]),
                     (0, 255, 255), 2)

            # 方向检测和统计更新
            for obj in result['detections']:
                direction = direction_detector.detect_direction(traffic_counter, obj['track_id'])

                # 更新流量统计
                vehicle_type = obj['type']
                with traffic_data_lock:
                    if direction != "unknown" and direction != "stationary":
                        traffic_data['direction_counts'][vehicle_type][direction] += 1

                    # 更新历史计数
                    traffic_data['count_history'][current_time][vehicle_type] += 1
                    traffic_data['total_counts'][vehicle_type] += 1

            # 显示当前帧的车辆计数
            current_counts = defaultdict(int)
            for obj in result['detections']:
                vehicle_type = obj['type']
                current_counts[vehicle_type] += 1

            y_pos = 60
            for vehicle, count in current_counts.items():
                cv2.putText(display_frame, f"{vehicle}: {count}", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_pos += 30

            # 显示总流量统计
            with traffic_data_lock:
                y_pos = height - 150
                cv2.putText(display_frame, "累计流量统计:", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 30
                for vehicle, count in traffic_data['total_counts'].items():
                    cv2.putText(display_frame, f"{vehicle}: {count}", (20, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_pos += 25

            # 显示FPS
            cv2.putText(display_frame, f"FPS: {real_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 显示报告提示
            cv2.putText(display_frame, "按 'r' 生成报告", (width - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

            cv2.imshow("车辆检测与流量统计", display_frame)

            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋👋👋👋👋👋👋👋 用户退出")
                break
            elif key == ord('r'):
                report_info = generate_report()
                print(f"\n📊📊📊📊 用户请求报告: {report_info}")
                # 重置历史数据但保留总计数
                with traffic_data_lock:
                    traffic_data['count_history'] = defaultdict(lambda: defaultdict(int))

    except Exception as e:

        print(f"❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌ 错误: {str(e)}")

        if 'frame' in locals():
            cv2.imwrite("error_frame.jpg", frame)

        # 发生错误时也捕获当前车流量

        with traffic_data_lock:

            final_counts = dict(traffic_data['total_counts'])

    finally:

        # 退出前生成最终报告

        final_report = generate_report()

        print(f"\n📑📑📑📑📑📑📑📑 最终报告: {final_report}")

        # 确保获取最终车流量数据

        if final_counts is None:
            with traffic_data_lock:
                final_counts = dict(traffic_data['total_counts'])

        running = False

        cap.release()

        cv2.destroyAllWindows()

    return final_counts  # 返回最终车流量统计


if __name__ == "__main__":

    traffic_counts = main()

    print("\n最终车流量统计:")

    for vehicle_type, count in traffic_counts.items():
        print(f"{vehicle_type}: {count}")