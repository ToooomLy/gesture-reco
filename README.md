# 手势识别项目

基于 MediaPipe + RealSense 深度相机 + SVM 的实时手势识别系统。

## 功能

- 使用 RealSense D455 相机采集手部关键点数据
- 提取 63 维归一化手部特征
- 使用 SVM 训练手势分类器
- 实时识别 `rock`、`paper`、`scissors`、`thumb_up`、`ok`、`none` 六种手势

## 环境要求

- Python 3.10+
- RealSense D455 深度相机（或其他支持 pyrealsense2 的型号）

## 安装依赖

```bash
pip install -r requirements.txt
```

依赖列表：

- mediapipe
- pyrealsense2
- opencv-python
- numpy
- scikit-learn
- joblib
- pandas

## 项目结构

```text
.
├── hand_landmarker.task      # MediaPipe 手部关键点检测模型
├── gesture_recognizer.task   # MediaPipe 手势识别模型（备用）
├── gesture_svm.joblib        # 训练好的 SVM 分类器
├── dataset.csv               # 采集的数据集
├── features.py               # 手部特征提取
├── collect.py                # 数据采集程序
├── train.py                  # 模型训练程序
├── demo.py                   # 实时识别演示
├── requirements.txt          # 依赖列表
└── README.md                 # 说明文档
```

## 使用流程

### 1. 采集数据

运行数据采集程序：

```bash
python collect.py
```

窗口会显示实时画面，按以下按键操作：

| 按键 | 功能 |
|------|------|
| `1` | 选择标签 `rock` |
| `2` | 选择标签 `paper` |
| `3` | 选择标签 `scissors` |
| `4` | 选择标签 `thumb_up` |
| `5` | 选择标签 `ok` |
| `0` | 选择标签 `none` |
| `SPACE` | 开始 / 暂停采集 |
| `s` | 完成当前动作采集并清空标签 |
| `q` | 退出程序 |

**采集步骤：**

1. 按数字键选择要采集的手势标签
2. 按 `SPACE` 开始采集
3. 在摄像头前做出对应手势
4. 按 `SPACE` 暂停，或按 `s` 完成当前手势
5. 重复步骤 1-4，采集其他手势
6. 按 `q` 退出

**提示：**

- 必须先选择标签，再按空格，否则会有提示
- 暂停后继续采集会保留之前的计数，不会清零
- 切换标签时会自动停止当前采集，避免数据混淆

### 2. 训练模型

采集完数据后，运行训练程序：

```bash
python train.py
```

程序会读取 `dataset.csv`，训练 SVM 分类器，并保存为 `gesture_svm.joblib`。

### 3. 实时识别

训练完成后，运行演示程序：

```bash
python demo.py
```

程序会实时显示当前识别到的手势和置信度。按 `q` 退出。

## 特征说明

[features.py](features.py) 中提取的 63 维特征包括：

1. 以手腕为原点进行平移
2. 建立手掌局部坐标系（基于手腕、食指根部、中指根部、小指根部）
3. 将 21 个手部关键点转换到手掌坐标系
4. 按手掌尺寸进行尺度归一化
5. 输出 21 × 3 = 63 维特征向量

## 当前数据集统计

| 手势 | 样本数 | 占比 |
|------|--------|------|
| rock | 6,060 | 33.7% |
| none | 4,099 | 22.8% |
| paper | 2,053 | 11.4% |
| thumb_up | 1,973 | 11.0% |
| ok | 1,892 | 10.5% |
| scissors | 1,882 | 10.5% |

**总计：** 17,959 条样本，63 维特征

> 注：数据分布不太均衡，`rock` 和 `none` 样本较多。如果某些手势识别率较低，可以适当补充对应样本。

## 注意事项

1. 确保 RealSense 相机已连接并正常工作
2. 采集时尽量保证手部在画面中央，光线充足
3. 每个手势建议采集 500~2000 条样本
4. 训练前可以检查 `dataset.csv` 的数据分布，必要时补充样本
5. 如果 Qt 字体警告不影响程序运行，可以忽略

## 许可证

本项目仅供学习和研究使用。
