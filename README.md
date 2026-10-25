基于YOLOv8的无人机影像光伏板缺陷系统
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/01b8e5d1442844f3ae35e97c5ad3266c.png)

**核心优势：
1⃣️前沿技术：采用YOLOv8深度学习模型，检测精度高、速度快，轻松识别光伏板缺陷目标！
2⃣️功能齐全：含完整训练代码+数据集（1819张训练集+227张验证集+227张测试集），种类分为[‘Hotspot’, ‘PID’, ‘Shadow’, ‘String’]，支持自定义数据训练；图像/视频/摄像头实时检测一键搞定！
3⃣️傻瓜式操作：Python + PySide6开发的可视化界面，参数调节、结果展示一目了然，**


以下文字及代码仅供参考学习。
---

## 📁 一、项目目录结构

```
Drone_PV_Defect_Detection/
├── main.py                      # 主程序入口
├── ui_mainwindow.py             # PySide6 UI 文件（由 Designer 生成）
├── pv_detection.py              # 核心检测逻辑
├── models/                      # 训练好的模型
│   └── best.pt                  # 最佳权重文件
├── dataset/                     # 数据集图片
│   ├── train/                   # 1819 张
│   ├── val/                     # 227 张
│   └── test/                    # 227 张
├── results/                     # 检测结果输出
└── requirements.txt             # Python 依赖
```

---

## 🔧 二、requirements.txt

```txt
python==3.9
torch==2.0.1+cu118
torchvision==0.15.2+cu118
opencv-python==4.8.0
PySide6==6.5.0
ultralytics==8.0.0
numpy==1.24.0
Pillow==9.5.0
```

> 安装命令：
```bash
pip install -r requirements.txt
```

---

## 🧩 三、数据集信息（已标注）

| 类别 | 中文含义 |
|------|----------|
| Hotspot | 热斑（局部过热） |
| PID | 电势诱导衰减（Potential Induced Degradation） |
| Shadow | 阴影遮挡 |
| String | 串间断路或连接异常 |

> ✅ 数据格式：YOLO 格式 `.txt`，每张图像对应一个标签文件  
> ✅ 总图像数：**1819 + 227 + 227 = 2273 张**

---

## 🚀 四、YOLOv8 训练代码（`train_pv.py`）

```python
# train_pv.py
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 开始训练
results = model.train(
    data='pv.yaml',               # 数据集配置文件
    epochs=150,
    imgsz=640,                    # 输入尺寸
    batch=16,                     # 批大小
    name='panel_yolo8s',          # 实验名称
    device=0,                     # GPU ID
    patience=30,                  # 早停
    save=True,
    plots=True
)

# 验证测试集性能
metrics = model.val(data='pv.yaml', split='test')
print(f"mAP@0.5: {metrics.box.map50:.4f}, mAP@0.5:0.95: {metrics.box.map:.4f}")
```

---

### ✅ `pv.yaml` 配置文件

```yaml
# pv.yaml
path: ./dataset
train: images/train
val: images/val
test: images/test

nc: 4
names: ['Hotspot', 'PID', 'Shadow', 'String']
```

---阴影遮挡',
'组串异常'

## 🖼️ 五、核心检测模块：`pv_detection.py`

```python
# pv_detection.py
import cv2
import numpy as np
from ultralytics import YOLO
import os

class PVDefectDetector:
    def __init__(self, model_path='models/best.pt', device='cpu'):
        self.model = YOLO(model_path)
        self.device = device
        self.names = ['Hotspot', 'PID', 'Shadow', 'String']

    def detect_image(self, image_path, conf_threshold=0.25, iou_threshold=0.45):
        """检测单张图片"""
        results = self.model(image_path, conf=conf_threshold, iou=iou_threshold)
        return results[0]

    def detect_video(self, video_path, conf_threshold=0.25, iou_threshold=0.45):
        """检测视频文件"""
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame, conf=conf_threshold, iou=iou_threshold)
            annotated_frame = results[0].plot()
            yield annotated_frame
        cap.release()

    def detect_camera(self, conf_threshold=0.25, iou_threshold=0.45):
        """调用摄像头实时检测"""
        cap = cv2.VideoCapture(0)  # 使用默认摄像头
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame, conf=conf_threshold, iou=iou_threshold)
            annotated_frame = results[0].plot()
            yield annotated_frame
        cap.release()
```

---

## 🎨 六、主程序界面（PySide6）：`main.py`

```python
# main.py
import sys
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QLineEdit, QSlider,
    QComboBox, QCheckBox, QLabel, QTextEdit, QGroupBox
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QFont
from PySide6.QtCore import Qt, QTimer, QSize
from pv_detection import PVDefectDetector
import cv2
import numpy as np

class PVDefectSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv8无人机影像光伏缺陷检测系统")
        self.setGeometry(100, 100, 1200, 800)

        self.detector = None
        self.current_image = None
        self.results = None
        self.init_ui()
        self.load_model()

    def init_ui(self):
        # 设置主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # 左侧：图像显示区
        self.image_label = QLabel("点击加载图像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: white;")
        self.image_label.setMinimumSize(800, 600)

        # 右侧：控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        # 模型选择
        self.model_path = QLineEdit("/path/to/models/best.pt")
        self.btn_load_model = QPushButton("加载模型")
        self.btn_load_model.clicked.connect(self.load_model)

        # 资源选择
        self.btn_open_image = QPushButton("打开图片")
        self.btn_open_image.clicked.connect(self.open_image)
        self.btn_open_video = QPushButton("打开视频")
        self.btn_open_video.clicked.connect(self.open_video)
        self.btn_open_camera = QPushButton("打开摄像头")
        self.btn_open_camera.clicked.connect(self.open_camera)

        # 参数设置
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(25)
        self.conf_label = QLabel("Conf: 0.25")

        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(0, 100)
        self.iou_slider.setValue(45)
        self.iou_label = QLabel("IOU: 0.45")

        self.conf_slider.valueChanged.connect(self.update_conf)
        self.iou_slider.valueChanged.connect(self.update_iou)

        # 检测结果展示
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)

        # 保存按钮
        self.btn_save = QPushButton("保存结果")
        self.btn_save.clicked.connect(self.save_results)

        # 添加到布局
        control_layout.addWidget(self.model_path)
        control_layout.addWidget(self.btn_load_model)
        control_layout.addWidget(self.btn_open_image)
        control_layout.addWidget(self.btn_open_video)
        control_layout.addWidget(self.btn_open_camera)
        control_layout.addWidget(self.conf_label)
        control_layout.addWidget(self.conf_slider)
        control_layout.addWidget(self.iou_label)
        control_layout.addWidget(self.iou_slider)
        control_layout.addWidget(self.result_box)
        control_layout.addWidget(self.btn_save)

        layout.addWidget(self.image_label)
        layout.addWidget(control_panel)

    def load_model(self):
        try:
            model_path = self.model_path.text()
            self.detector = PVDefectDetector(model_path=model_path)
            print("✅ 模型加载成功！")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png)")
        if file_path:
            self.current_image = file_path
            self.display_image(file_path)

    def open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            self.video_capture = cv2.VideoCapture(file_path)
            self.process_video()

    def process_video(self):
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            results = self.detector.detect_image(frame, self.conf_threshold, self.iou_threshold)
            annotated_frame = results.plot()
            self.display_image_from_array(annotated_frame)

    def open_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_frame)
        self.timer.start(30)

    def update_camera_frame(self):
        ret, frame = self.cap.read()
        if ret:
            results = self.detector.detect_image(frame, self.conf_threshold, self.iou_threshold)
            annotated_frame = results.plot()
            self.display_image_from_array(annotated_frame)

    def display_image(self, path):
        pixmap = QPixmap(path)
        self.image_label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio))

    def display_image_from_array(self, array):
        h, w = array.shape[:2]
        bytes_per_line = 3 * w
        qimage = QImage(array.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio))

    def update_conf(self):
        self.conf_threshold = self.conf_slider.value() / 100.0
        self.conf_label.setText(f"Conf: {self.conf_threshold:.2f}")

    def update_iou(self):
        self.iou_threshold = self.iou_slider.value() / 100.0
        self.iou_label.setText(f"IOU: {self.iou_threshold:.2f}")

    def save_results(self):
        if self.current_image:
            result_img = self.detector.detect_image(self.current_image, self.conf_threshold, self.iou_threshold)
            annotated_img = result_img.plot()
            output_dir = "results"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(self.current_image))
            cv2.imwrite(output_path, annotated_img)
            print(f"✅ 结果已保存至: {output_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PVDefectSystem()
    window.show()
    sys.exit(app.exec())
```

---

## ✅ 七、功能亮点总结

| 功能 | 描述 |
|------|------|
| ✅ 支持多源输入 | 图片、视频、摄像头 |
| ✅ 实时检测 | 摄像头画面实时标注 |
| ✅ 参数调节 | 自定义置信度、IoU 阈值 |
| ✅ 结果可视化 | 目标数量、耗时、类别清晰展示 |
| ✅ 结果保存 | 自动保存带框标注图像 |
| ✅ 易于扩展 | 可替换模型、添加新类 |

---


