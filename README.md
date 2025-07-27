智能交通监控系统
项目概述
这是一个基于深度学习的智能交通监控系统，集成了车辆检测、流量统计、车牌识别和违章检测功能。系统使用YOLOv5进行目标检测，结合SORT跟踪算法，实现了高效的道路交通分析。
运行环境
硬件要求
CPU: 四核处理器或更高
RAM: 8GB或更高
支持CUDA的GPU（可选但推荐）
软件环境
​​操作系统​​: Linux (推荐Ubuntu 20.04+) / Windows 10+
​​Python​​: 3.8+
​​CUDA​​ (GPU用户): 11.1+
​​cuDNN​​ (GPU用户): 8.0.5+

Python依赖
pip install torch torchvision torchaudio
pip install ultralytics opencv-python numpy pandas matplotlib requests
pip install paddlepaddle paddleocr
pip install easyocr

安装步骤

1.克隆项目仓库：

git clone https://github.com/yourusername/traffic-monitoring-system.git

cd traffic-monitoring-system

2.安装Python依赖
pip install -r requirements.txt

3.下载预训练模型：
# YOLOv5模型
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt -P models/

# PaddleOCR模型
mkdir -p models/Paddle
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar -P models/Paddle
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar -P models/Paddle
启动指南

车辆检测与流量统计
python traffic_flow/Traffic_flow.py
按 q 退出程序
按 r 生成交通报告

车牌识别
python ocr/Ocr.py

违章检测
python violation/Violation.py
