# 🚦 智能交通监控系统

一个基于深度学习的多功能智能交通监控平台，集成了 **车辆检测**、**流量统计**、**车牌识别** 以及 **交通违章检测** 等模块。系统采用 YOLOv5 目标检测模型与 SORT 跟踪算法，结合 OCR 技术，实现对城市交通状况的高效感知与分析。

---

## 📌 项目特色

- ✅ 支持多类别目标识别（轿车、货车、行人、非机动车等）
- ✅ 实时交通流量监测与统计分析
- ✅ 精准车牌识别（中英文支持）
- ✅ 交通违章行为检测（逆行、压线等）
- ✅ 自动生成交通报告（支持可视化与导出）

---

## 🧱 项目结构

traffic-monitoring-system/
├── models/ # 预训练模型存放目录
├── traffic_flow/ # 流量统计模块
│ └── Traffic_flow.py
├── ocr/ # 车牌识别模块（支持PaddleOCR/EasyOCR）
│ └── Ocr.py
├── violation/ # 交通违章检测模块
│ └── Violation.py
├── utils/ # 常用工具函数（绘图、解码、日志等）
├── config/ # 配置文件（如摄像头参数等）
├── requirements.txt # 所有依赖包列表
└── README.md # 项目说明文档

---

## 💻 环境要求

### ✅ 硬件推荐

- CPU：四核及以上
- 内存：8GB+
- GPU（推荐，提升检测速度）：支持 CUDA 的 NVIDIA 显卡

### ✅ 软件环境

| 软件        | 推荐版本        |
|-------------|-----------------|
| 操作系统    | Ubuntu 20.04+ / Windows 10+ |
| Python      | 3.8 及以上       |
| CUDA（可选）| 11.1+           |
| cuDNN（可选）| 8.0.5+         |

---

## 📦 依赖安装

> 强烈建议使用虚拟环境（如 `venv` 或 `conda`）

```bash
# 克隆项目
git clone https://github.com/yourusername/traffic-monitoring-system.git
cd traffic-monitoring-system

# 安装 Python 依赖
pip install -r requirements.txt

# 若未包含 requirements.txt，可手动安装：
pip install torch torchvision torchaudio
pip install ultralytics opencv-python numpy pandas matplotlib requests
pip install paddlepaddle paddleocr
pip install easyocr
```

## 🔍 模型下载

### ✅ YOLOv5 轻量级模型（目标检测）：
```bash
mkdir -p models
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt -P models/
```
### ✅PaddleOCR 模型（车牌识别）：
```bash
mkdir -p models/Paddle
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar -P models/Paddle
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar -P models/Paddle
```
解压：
```bash
cd models/Paddle
tar -xf ch_PP-OCRv3_det_infer.tar
tar -xf ch_PP-OCRv3_rec_infer.tar
```
## 🚀 启动方式

### ▶ 1. 车辆检测与流量统计
```bash
python traffic_flow/Traffic_flow.py
```
按 q 退出程序
按 r 生成交通统计报告

### ▶ 2. 车牌识别
```bash
python ocr/Ocr.py
```
### ▶ 3. 交通违章检测
```bash
python violation/Violation.py
```
## 🙌 致谢
YOLOv5 by Ultralytics

PaddleOCR

EasyOCR
