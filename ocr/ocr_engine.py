import os
import cv2
import numpy as np
import time
import platform  # 添加平台检测库
from ocr.ocr_utils import preprocess_image, postprocess_text
from ocr.plate_cropper import PlateCropper

# 检查是否安装PaddleOCR
try:
    from paddleocr import PaddleOCR

    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

# 添加EasyOCR作为备选方案
try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# 在PlateRecognizer初始化前设置
os.environ['USE_ONNX'] = '1'  # 强制启用ONNX

class PlateRecognizer:
    def __init__(self, use_gpu=False, lang='ch', model_dir=None):
        """
        针对全志H618优化的车牌识别引擎
        :param use_gpu: 是否使用GPU (全志H618上设为False)
        :param lang: 识别语言
        :param model_dir: 模型目录路径
        """
        self.ocr_engine = None
        self.cropper = PlateCropper()
        self.ocr_type = None  # 记录使用的OCR类型
        self.use_onnx = False  # 是否使用ONNX加速

        # 获取系统架构信息
        self.arch = platform.machine()
        self.is_arm = self.arch.startswith('aarch') or self.arch.startswith('arm')

        # 模型初始化标志
        self.models_initialized = False

        # 初始化OCR引擎
        self.init_ocr_engine(use_gpu, lang, model_dir)

        # 如果PaddleOCR不可用，尝试EasyOCR
        if not self.ocr_engine and EASYOCR_AVAILABLE:
            self.init_easyocr_engine(use_gpu, lang)

        # 如果都没有，使用基础OCR
        if not self.ocr_engine:
            print("⚠️ 警告: PaddleOCR和EasyOCR均不可用，使用简易字符分割方法")
            self.ocr_type = "basic"

    def init_ocr_engine(self, use_gpu, lang, model_dir):
        """初始化PaddleOCR引擎"""
        if not PADDLEOCR_AVAILABLE:
            return

        try:
            # ARM架构优化配置
            if self.is_arm:
                os.environ['OMP_NUM_THREADS'] = '1'
                os.environ['KMP_BLOCKTIME'] = '1'
                os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'

            # 配置参数
            config_params = {
                'use_angle_cls': False,
                'lang': lang,
                'use_gpu': use_gpu,
                'rec_image_shape': "3, 36, 320",
                'det_limit_side_len': 480,
                'det_db_thresh': 0.3,
                'det_db_box_thresh': 0.4,
                'det_db_unclip_ratio': 1.5,
                'max_batch_size': 1,
                'use_tensorrt': False,
                'enable_mkldnn': False  # ARM上禁用MKL-DNN
            }

            # 模型路径设置
            if model_dir:
                # 优先使用更轻量的PP-OCRv2模型
                v2_rec = os.path.join(model_dir, 'ch_PP-OCRv2_rec_infer')
                v2_det = os.path.join(model_dir, 'ch_PP-OCRv2_det_infer')

                if os.path.exists(v2_rec) and os.path.exists(v2_det):
                    config_params['rec_model_dir'] = v2_rec
                    config_params['det_model_dir'] = v2_det
                else:
                    # 回退到v3模型
                    v3_rec = os.path.join(model_dir, 'ch_PP-OCRv3_rec_infer')
                    v3_det = os.path.join(model_dir, 'ch_PP-OCRv3_det_infer')

                    if os.path.exists(v3_rec):
                        config_params['rec_model_dir'] = v3_rec
                    if os.path.exists(v3_det):
                        config_params['det_model_dir'] = v3_det

            # ARM架构特殊处理
            if self.is_arm:
                # 尝试启用ONNX推理
                if model_dir:
                    onnx_path = os.path.join(model_dir, 'onnx_models')
                    if os.path.exists(onnx_path):
                        config_params['use_onnx'] = True
                        config_params['onnx_model_path'] = onnx_path
                        self.use_onnx = True

            # 创建OCR引擎
            self.ocr_engine = PaddleOCR(**config_params)
            self.models_initialized = True
            self.ocr_type = "paddle"
            print(f"✅ PaddleOCR引擎初始化成功 (架构: {self.arch})")
            if self.use_onnx:
                print("   - 使用ONNX加速推理")
        except Exception as e:
            print(f"❌ PaddleOCR初始化失败: {str(e)}")
            self.ocr_engine = None

    def init_easyocr_engine(self, use_gpu, lang):
        """初始化EasyOCR引擎作为备选"""
        try:
            # EasyOCR配置
            self.ocr_engine = easyocr.Reader(
                ['ch_sim', 'en'],
                gpu=use_gpu,
                model_storage_directory='./easyocr_models',
                download_enabled=True
            )
            self.ocr_type = "easyocr"
            print(f"✅ EasyOCR引擎初始化成功 (备选方案)")
            return True
        except Exception as e:
            print(f"EasyOCR初始化失败: {str(e)}")
            self.ocr_engine = None
            return False

    def recognize_plate(self, vehicle_img, vehicle_bbox):
        """
        识别车辆图像中的车牌
        :param vehicle_img: 完整图像帧
        :param vehicle_bbox: 车辆边界框 (x1, y1, x2, y2)
        :return: (车牌号, 车牌边界框, 置信度)
        """
        # 裁剪车辆区域
        x1, y1, x2, y2 = vehicle_bbox
        vehicle_roi = vehicle_img[int(y1):int(y2), int(x1):int(x2)]

        # 从车辆区域中裁剪车牌
        plate_img, plate_bbox = self.cropper.crop_plate_from_vehicle(vehicle_roi, vehicle_bbox)

        # 如果车牌区域太小，直接返回空结果
        if plate_img.size == 0 or plate_img.shape[0] < 20 or plate_img.shape[1] < 40:
            return "", (0, 0, 0, 0), 0.0

        # 预处理图像
        processed_img = preprocess_image(plate_img)

        # 根据OCR引擎类型选择识别方法
        if self.ocr_type == "paddle":
            return self._recognize_with_paddleocr(processed_img, plate_bbox, (x1, y1))
        elif self.ocr_type == "easyocr":
            return self._recognize_with_easyocr(plate_img, plate_bbox, (x1, y1))
        else:
            return self._recognize_with_basic_ocr(processed_img, plate_bbox, (x1, y1))

    def _recognize_with_paddleocr(self, plate_img, plate_bbox, offset):
        """
        使用PaddleOCR识别车牌
        """
        try:
            # 执行OCR
            start_time = time.time()
            result = self.ocr_engine.ocr(plate_img, cls=False)  # 禁用方向分类

            # 解析结果
            plate_text = ""
            confidence = 0.0
            if result and result[0]:
                # 获取所有文本和置信度
                texts = [line[1][0] for line in result[0]]
                confidences = [line[1][1] for line in result[0]]

                # 合并所有文本（车牌通常只有一行）
                plate_text = ''.join(texts)

                # 计算平均置信度
                if confidences:
                    confidence = sum(confidences) / len(confidences)

            # 后处理文本
            plate_text = postprocess_text(plate_text)

            # 调整车牌边界框坐标到原图
            px1, py1, px2, py2 = plate_bbox
            abs_bbox = (px1 + offset[0], py1 + offset[1], px2 + offset[0], py2 + offset[1])

            # 计算处理时间
            process_time = time.time() - start_time
            print(f"OCR处理时间: {process_time:.3f}s, 识别结果: {plate_text}")

            return plate_text, abs_bbox, confidence
        except Exception as e:
            print(f"PaddleOCR识别异常: {str(e)}")
            return "", (0, 0, 0, 0), 0.0

    def _recognize_with_easyocr(self, plate_img, plate_bbox, offset):
        """
        使用EasyOCR识别车牌
        """
        try:
            # 执行OCR
            start_time = time.time()
            results = self.ocr_engine.readtext(plate_img)

            # 解析结果
            plate_text = ""
            confidence = 0.0

            if results:
                # 获取所有文本和置信度
                texts = [result[1] for result in results]
                confidences = [result[2] for result in results]

                # 合并所有文本
                plate_text = ''.join(texts)

                # 计算平均置信度
                if confidences:
                    confidence = sum(confidences) / len(confidences)

            # 后处理文本
            plate_text = postprocess_text(plate_text)

            # 调整车牌边界框坐标到原图
            px1, py1, px2, py2 = plate_bbox
            abs_bbox = (px1 + offset[0], py1 + offset[1], px2 + offset[0], py2 + offset[1])

            # 计算处理时间
            process_time = time.time() - start_time
            print(f"EasyOCR处理时间: {process_time:.3f}s, 识别结果: {plate_text}")

            return plate_text, abs_bbox, confidence
        except Exception as e:
            print(f"EasyOCR识别异常: {str(e)}")
            return "", (0, 0, 0, 0), 0.0

    def _recognize_with_basic_ocr(self, plate_img, plate_bbox, offset):
        """
        简易OCR方法 (OCR引擎不可用时使用)
        """
        # 转换为灰度图
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()

        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 筛选字符轮廓
        char_contours = []
        height, width = plate_img.shape[:2]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # 过滤尺寸不合理的轮廓
            if w < 5 or h < 15 or w > width * 0.5 or h < height * 0.3:
                continue

            char_contours.append((x, y, w, h))

        # 按x坐标排序
        char_contours = sorted(char_contours, key=lambda c: c[0])

        # 识别字符 (简化版)
        plate_text = ""
        for i, (x, y, w, h) in enumerate(char_contours):
            # 裁剪字符区域
            char_img = binary[y:y + h, x:x + w]

            # 调整大小
            char_img = cv2.resize(char_img, (20, 40))

            # 这里可以添加简单的模板匹配或CNN识别
            # 简化处理：返回位置信息
            plate_text += "X"  # 用X代替识别字符

        # 后处理文本
        plate_text = postprocess_text(plate_text)

        # 调整车牌边界框坐标到原图
        px1, py1, px2, py2 = plate_bbox
        abs_bbox = (px1 + offset[0], py1 + offset[1], px2 + offset[0], py2 + offset[1])

        return plate_text, abs_bbox, 0.5  # 固定置信度