import cv2
import numpy as np


def preprocess_image(image, target_height=36):
    """
    针对H618优化图像预处理，增强前两个字符识别能力。
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # === 图像增强 - 锐化处理 ===
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)

    # 自适应阈值
    binary = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 21, 5
    )

    # 形态学降噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # === 加权强调车牌左侧（前2字符）区域亮度 ===
    h, w = cleaned.shape
    mask = np.ones_like(cleaned, dtype=np.uint8) * 255
    mask[:, int(w * 0.3):] = 100  # 降低后半部分影响
    cleaned = cv2.addWeighted(cleaned, 1.2, mask, -0.2, 0)

    # 尺寸归一化
    scale = target_height / float(h)
    target_width = int(w * scale)
    resized = cv2.resize(cleaned, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    # 转为三通道
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    return resized


def postprocess_text(text):
    """
    车牌识别后处理（强化前两位规则）
    """
    # 基本清洗
    cleaned = ''.join(c for c in text if c.isalnum() or '\u4e00' <= c <= '\u9fff')

    # 替换常见误识别
    char_map = {
        'O': '0', 'o': '0', 'Q': '0', 'D': '0',
        'I': '1', 'l': '1', '|': '1',
        'Z': '2',
        'B': '8',
        'S': '5', '$': '5',
        'U': 'V', 'v': 'V', 'W': 'V'
    }

    corrected = ''.join(char_map.get(c, c) for c in cleaned)

    # === 特殊逻辑：修复第一个字符和第二字符 ===
    provinces = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
    default_province = "粤"

    if len(corrected) >= 1:
        # 第一位应为汉字
        if not ('\u4e00' <= corrected[0] <= '\u9fff'):
            # 如果误识别为拉丁字母，可能是某些汉字类似形
            corrected = default_province + corrected[1:]

    if len(corrected) >= 2:
        # 第二位应为大写英文字母
        if not corrected[1].isalpha():
            corrected = corrected[0] + 'A' + corrected[2:]

    # 只保留前7位（车牌通常为7位）
    final_plate = corrected[:7]

    return final_plate.upper()