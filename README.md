# 基于dlib模型的人脸分析系统

本项目是一个综合性的人脸分析系统，集成了人脸识别和情绪检测功能。它利用dlib模型和opencv技术，可实现实时分析面部特征和部分表情。

## 项目结构

```
face-analysis-system
├── src
│   │── face_recognition.py
│   │── emotion_recognition.py
|   |—— student_monitor.py
|   |—— Expression_Calibration.py
|   |—— Calibrating_camera.py
│   │── utils.py
│   └── app.py
├── model
│   ├── shape_predictor_68_face_landmarks.dat
│   └── dlib_face_recognition_resnet_model_v1.dat
├── data
│   └── face_database.pkl
├── tests
│   ├── __init__.py
│   ├── test_face_recognition.py
│   └── test_emotion_recognition.py
├── docs
│   ├── setup.md
│   └── usage.md
├── requirements.txt
├── main.py
└── README.md
```

## 功能特点

- **人脸识别**: 使用基于dlib预训练模型检测和识别人脸。
- **情绪检测**: 分析面部表情特征点以及头部姿态以确定情绪状态（五种：专注，分心，困惑，疲惫，兴奋）。
- **实时处理**: 利用摄像头输入进行实时视频流分析。

## 安装步骤

1. 克隆仓库：
   ```
   git clone -b dlib https://github.com/maoluois/Facial_detection
   cd face-analysis-system
   ```

2. 安装必要的依赖：
   ```
   pip install -r requirements.txt
   ```
   需要安装pytorch，cpu即可，建议使用python3.10 或 anaconda环境来安装这些依赖

3. 载必要的模型并将它们放在model目录中。
   ```
   https://github.com/davisking/dlib-models
   ```

## Usage

请先安装好上面提到的依赖和环境 要运行应用程序，直接使用编译器运行app.py
或者打开终端用cd指令进入文件根目录 并执行 python3 ./src/app.py

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.