import cv2
import numpy as np
import time  # 新增导入用于调试
import os  # 添加缺失的os导入


class PlateCropper:
    def __init__(self, plate_aspect_ratio=(3.5, 1), margin_ratio=0.05):
        """
        针对全志H618优化的车牌裁剪器
        :param plate_aspect_ratio: 车牌宽高比 (宽, 高)
        :param margin_ratio: 裁剪时保留的边界比例
        """
        self.plate_aspect_ratio = plate_aspect_ratio
        self.margin_ratio = margin_ratio
        # 预编译的形态学操作内核
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # ===== 新增：扩展颜色范围 =====
        # 修改 HSV 范围，更宽容反光或光照差异
        self.blue_lower = np.array([90, 50, 50])
        self.blue_upper = np.array([130, 255, 255])

        self.green_lower = np.array([35, 40, 40])
        self.green_upper = np.array([90, 255, 255])

        self.yellow_lower = np.array([15, 40, 130])
        self.yellow_upper = np.array([40, 255, 255])

        self.white_lower = np.array([0, 0, 190])
        self.white_upper = np.array([180, 60, 255])

        self.black_lower = np.array([0, 0, 0])
        self.black_upper = np.array([180, 255, 70])

    def is_plate_color(self, roi):
        """
        检测区域是否包含车牌颜色（蓝牌、绿牌、黄牌、白牌、黑牌）
        :param roi: 候选区域图像 (BGR格式)
        :return: 是否符合车牌颜色特征
        """
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 创建各种颜色掩码
        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)  # 新增
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)  # 新增
        black_mask = cv2.inRange(hsv, self.black_lower, self.black_upper)  # 新增

        # 合并颜色掩码
        color_mask = cv2.bitwise_or(blue_mask, green_mask)
        color_mask = cv2.bitwise_or(color_mask, yellow_mask)  # 新增
        color_mask = cv2.bitwise_or(color_mask, white_mask)  # 新增
        color_mask = cv2.bitwise_or(color_mask, black_mask)  # 新增

        # 计算颜色区域比例
        color_pixels = cv2.countNonZero(color_mask)
        total_pixels = roi.shape[0] * roi.shape[1]

        # 降低阈值到20%（原为30%）
        return color_pixels / total_pixels > 0.2

    def crop_plate_from_vehicle(self, vehicle_img, vehicle_bbox):
        """
        从车辆图像中裁剪车牌区域
        :param vehicle_img: 车辆图像 (RGB格式)
        :param vehicle_bbox: 车辆边界框 (x1, y1, x2, y2)
        :return: 裁剪后的车牌图像, 车牌边界框坐标
        """
        # 获取车辆区域尺寸
        vehicle_height, vehicle_width = vehicle_img.shape[:2]

        # 调试信息
        debug_dir = "plate_cropper_debug"
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = int(time.time())

        # 针对全志H618优化: 降低分辨率处理
        if vehicle_width > 640:
            scale = 640 / vehicle_width
            small_img = cv2.resize(vehicle_img, (640, int(vehicle_height * scale)))
            small_height, small_width = small_img.shape[:2]
        else:
            small_img = vehicle_img
            small_height, small_width = vehicle_height, vehicle_width
            scale = 1.0

        # ===== 新增：图像增强 =====
        # 转换为灰度图
        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)

        # 直方图均衡化增强对比度
        gray = cv2.equalizeHist(gray)

        # 保存预处理图像用于调试
        cv2.imwrite(f"{debug_dir}/gray_{timestamp}.jpg", gray)

        # ===== 新增：自适应边缘检测 =====
        # 计算图像平均灰度值
        gray_mean = np.mean(gray)

        # 动态设置Canny阈值
        low_thresh = max(30, gray_mean * 0.4)  # 原为80
        high_thresh = min(220, gray_mean * 2.0)  # 原为180

        # 边缘检测
        edges = cv2.Canny(gray, low_thresh, high_thresh, apertureSize=3, L2gradient=True)

        # 保存边缘检测结果
        cv2.imwrite(f"{debug_dir}/edges_{timestamp}.jpg", edges)

        # 形态学操作增强边缘
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 绘制轮廓用于调试
        contour_img = small_img.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(f"{debug_dir}/contours_{timestamp}.jpg", contour_img)

        # 筛选可能的车牌轮廓
        plate_contours = []
        for i, contour in enumerate(contours):
            # 计算轮廓的边界矩形
            x, y, w, h = cv2.boundingRect(contour)

            # 计算宽高比和相对尺寸
            aspect_ratio = w / float(h)
            rel_width = w / float(small_width)
            rel_height = h / float(small_height)

            # 放宽尺寸要求：从8%降到5%
            min_rel_size = 0.05

            # 放宽宽高比范围：从(1.5, 5.0)到(1.2, 6.0)
            if (1.2 < aspect_ratio < 6.0 and
                    min_rel_size < rel_width < 0.3 and
                    min_rel_size < rel_height < 0.3):

                # 检查颜色特征
                roi = small_img[y:y + h, x:x + w]
                if roi.size > 0 and self.is_plate_color(roi):
                    plate_contours.append((x, y, w, h))

                    # 保存候选区域用于调试
                    cv2.imwrite(f"{debug_dir}/candidate_{i}_{timestamp}.jpg", roi)

        # 如果没有找到轮廓，使用启发式方法定位车牌
        if not plate_contours:
            print(f"⚠️ 未找到轮廓，使用启发式定位 | 时间: {timestamp}")
            # 车牌通常在车辆下部1/3区域
            plate_y1 = int(small_height * 0.6)
            plate_y2 = small_height - 5
            plate_x1 = int(small_width * 0.1)
            plate_x2 = int(small_width * 0.9)

            # 调整回原始尺寸
            if scale != 1.0:
                plate_x1, plate_y1, plate_x2, plate_y2 = [int(coord / scale) for coord in
                                                          (plate_x1, plate_y1, plate_x2, plate_y2)]

            # 添加边界检查
            plate_y1 = max(0, min(plate_y1, vehicle_img.shape[0] - 1))
            plate_y2 = max(1, min(plate_y2, vehicle_img.shape[0]))
            plate_x1 = max(0, min(plate_x1, vehicle_img.shape[1] - 1))
            plate_x2 = max(1, min(plate_x2, vehicle_img.shape[1]))

            plate_img = vehicle_img[plate_y1:plate_y2, plate_x1:plate_x2]
            plate_bbox = (plate_x1, plate_y1, plate_x2, plate_y2)

            # 保存最终结果
            if plate_img.size > 0:
                cv2.imwrite(f"{debug_dir}/final_heuristic_{timestamp}.jpg", plate_img)
            return plate_img, plate_bbox

        # 选择面积最大的轮廓作为车牌
        plate_contour = max(plate_contours, key=lambda c: c[2] * c[3])
        x, y, w, h = plate_contour

        # 添加边界
        margin_x = int(w * self.margin_ratio)
        margin_y = int(h * self.margin_ratio)

        # 确保不超出图像边界
        y1 = max(0, y - margin_y)
        y2 = min(small_img.shape[0], y + h + margin_y)
        x1 = max(0, x - margin_x)
        x2 = min(small_img.shape[1], x + w + margin_x)

        # 裁剪车牌区域 (在原分辨率图像上)
        if scale != 1.0:
            x1, y1, x2, y2 = [int(coord / scale) for coord in (x1, y1, x2, y2)]

        # 添加边界检查
        y1 = max(0, min(y1, vehicle_img.shape[0] - 1))
        y2 = max(1, min(y2, vehicle_img.shape[0]))
        x1 = max(0, min(x1, vehicle_img.shape[1] - 1))
        x2 = max(1, min(x2, vehicle_img.shape[1]))

        plate_img = vehicle_img[y1:y2, x1:x2]
        plate_bbox = (x1, y1, x2, y2)

        # 保存最终结果
        if plate_img.size > 0:
            cv2.imwrite(f"{debug_dir}/final_{timestamp}.jpg", plate_img)

        return plate_img, plate_bbox