import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import os
import sys
import threading
import time
import random
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from emotion_recognition import EmotionRecognition
from face_recognition import FaceRecognition
from student_monitor import ClassroomMonitor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Expression_Calibration import ExpressionCalibration

plt.rcParams['font.sans-serif'] = ['SimHei']  
# 使用SimHei字体这样可以显示title中的中文，但这个不显示负号
plt.rcParams['axes.unicode_minus'] = False  
# 解决负号显示问题

class FaceAnalysisApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # 创建识别对象
        self.face_recognition = FaceRecognition()
        self.emotion_recognition = EmotionRecognition()
        
        # 创建标签页
        self.tab_control = ttk.Notebook(window)
        
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab2 = ttk.Frame(self.tab_control)
        self.tab3 = ttk.Frame(self.tab_control)

        
        self.tab_control.add(self.tab1, text='人脸注册与识别')
        self.tab_control.add(self.tab2, text='校准程序')
        self.tab_control.add(self.tab3, text='课堂状态监测')
        self.tab_control.pack(expand=1, fill="both")
        
        # 设置各个标签页的内容
        self.setup_registration_tab()
        self.setup_calibration_tab()
        self.setup_classroom_monitor_tab()
        
        # 视频捕获变量
        self.cap = None
        self.is_capturing = False
        
        # 定义关闭窗口时的行为
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 启动主循环
        self.window.mainloop()
    
    def setup_registration_tab(self):
        # 左侧: 图像显示和文件选择 
        left_frame = ttk.Frame(self.tab1)
        left_frame.pack(side="left", padx=10, pady=10)
        
        # 图像显示区域
        self.canvas = tk.Canvas(left_frame, width=400, height=300)
        self.canvas.pack(fill="both", expand=True)
    
        # 图像选择按钮和拍照按钮框架
        buttons_frame = ttk.Frame(left_frame)
        buttons_frame.pack(pady=5, fill="x")
    
        # 图像选择按钮
        btn_select = ttk.Button(buttons_frame, text="选择图像", command=self.select_image)
        btn_select.pack(side="left", padx=5, expand=True)
    
        # 拍照按钮
        btn_camera = ttk.Button(buttons_frame, text="摄像头拍照", command=self.capture_photo)
        btn_camera.pack(side="left", padx=5, expand=True)
        
        # 右侧: 注册和识别功能
        right_frame = ttk.Frame(self.tab1)
        right_frame.pack(side="right", padx=10, pady=10, fill="y")
        
        # 注册区域
        register_frame = ttk.LabelFrame(right_frame, text="注册人脸")
        register_frame.pack(pady=5, fill="x")
        
        ttk.Label(register_frame, text="人脸名称/ID:").pack(pady=5)
        self.face_name_var = tk.StringVar()
        ttk.Entry(register_frame, textvariable=self.face_name_var).pack(pady=5)
        ttk.Button(register_frame, text="注册", command=self.register_face).pack(pady=5)
        
        # 识别区域
        recognize_frame = ttk.LabelFrame(right_frame, text="识别人脸")
        recognize_frame.pack(pady=5, fill="x")
        ttk.Button(recognize_frame, text="识别", command=self.recognize_face).pack(pady=5)
        self.recognition_result_var = tk.StringVar()
        ttk.Label(recognize_frame, textvariable=self.recognition_result_var).pack(pady=5)
        
        # 数据库管理区域
        db_frame = ttk.LabelFrame(right_frame, text="人脸数据库管理")
        db_frame.pack(pady=5, fill="x")
        ttk.Button(db_frame, text="列出已注册人脸", command=self.list_faces).pack(pady=5)
        self.faces_list = tk.Text(db_frame, height=10, width=30)
        self.faces_list.pack(pady=5)
        ttk.Label(db_frame, text="要删除的人脸:").pack(pady=5)
        self.delete_name_var = tk.StringVar()
        ttk.Entry(db_frame, textvariable=self.delete_name_var).pack(pady=5)
        ttk.Button(db_frame, text="删除", command=self.delete_face).pack(pady=5)
    
        # 校准按钮
        calibration_btn = ttk.Button(register_frame, text="表情校准", command=self.start_expression_calibration)
        calibration_btn.pack(pady=5)

    def capture_photo(self):
        """打开摄像头并拍照"""
        # 创建一个顶层窗口用于显示摄像头预览
        camera_window = tk.Toplevel(self.window)
        camera_window.title("摄像头拍照")
        camera_window.geometry("640x520")
        
        # 创建显示摄像头预览的画布
        preview_canvas = tk.Canvas(camera_window, width=640, height=480)
        preview_canvas.pack(pady=5)
        
        # 创建拍照按钮
        photo_button = ttk.Button(camera_window, text="拍照", width=20)
        photo_button.pack(pady=5)
        
        # 开启摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            camera_window.destroy()
            return
        
        # 用于存储拍摄的照片
        captured_image = [None]
        
        def update_preview():
            ret, frame = cap.read()
            if ret:
                # 显示预览
                cv_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = PIL.Image.fromarray(cv_image_rgb)
                tk_image = PIL.ImageTk.PhotoImage(image=pil_image)
                preview_canvas.image = tk_image
                preview_canvas.create_image(320, 240, image=tk_image)
                
                # 如果窗口仍然存在，继续更新
                if camera_window.winfo_exists():
                    camera_window.after(10, update_preview)
                else:
                    cap.release()
        
        def take_photo():
            """拍照并关闭窗口"""
            ret, frame = cap.read()
            if ret:
                captured_image[0] = frame.copy()
                cap.release()
                camera_window.destroy()
                
                # 更新主界面上的图像
                self.current_image = captured_image[0]
                self.display_image(self.current_image, self.canvas)
        
        # 设置拍照按钮事件
        photo_button.config(command=take_photo)
        
        # 添加窗口关闭事件
        def on_window_close():
            cap.release()
            camera_window.destroy()
        
        camera_window.protocol("WM_DELETE_WINDOW", on_window_close)
        
        # 开始预览
        update_preview()
    
    def setup_calibration_tab(self):
        # 视频显示区域
        self.emotion_canvas = tk.Canvas(self.tab2, width=640, height=480)
        self.emotion_canvas.pack(pady=10)
        
        # 控制按钮
        control_frame = ttk.Frame(self.tab2)
        control_frame.pack(pady=5)
        
        self.start_calibration_btn = ttk.Button(control_frame, text="开始校准", command=self.start_calibration)
        self.start_calibration_btn.pack(side="left", padx=5)
        
        self.stop_calibration_btn = ttk.Button(control_frame, text="停止校准", command=self.stop_calibration)
        self.stop_calibration_btn.pack(side="left", padx=5)
        
        # AUs标准值校准说明
        info_frame = ttk.LabelFrame(self.tab2, text="AUs标准值校准")
        info_frame.pack(pady=10, fill="x", padx=10)
        
        # info不完整，还要完善 ！！！！！
        info_text = """
        请按照以下步骤进行校准：
        1. 皱眉 - 请尽量皱眉，保持3秒
        2. 上眼睑提升 - 请尽量睁大眼睛，保持3秒
        3. 眯眼 - 请尽量眯眼，保持3秒
        4. 嘴角上扬（微笑） - 请尽量微笑，保持3秒
        5. 嘴唇分离（轻微张口） - 请轻微张口，保持3秒
        6. 下颌下降（打哈欠） - 请尽量打哈欠，保持3秒
        """
        ttk.Label(info_frame, text=info_text, justify="left").pack(pady=5)
        
    def start_calibration(self):
        # 初始化校准数据
        self.calibration_data = {
            "head_forward": {"max": None, "min": None, "standard": None},
            "head_turn": {"max": None, "min": None, "standard": None},
            "frequent_movement": {"max": None, "min": None, "standard": None},
            "face_active": {"max": None, "min": None, "standard": None},
            "no_micro_expression": {"max": None, "min": None, "standard": None},
            "BrowFurrow": {"max": None, "min": None, "standard": None},
            "BrowFurrowAsymmetry": {"max": None, "min": None, "standard": None},
            "UpperLidRaiser": {"max": None, "min": None, "standard": None},
            "EyeSquint": {"max": None, "min": None, "standard": None},
            "Smile": {"max": None, "min": None, "standard": None},
            "LipsPart": {"max": None, "min": None, "standard": None},
            "JawDrop": {"max": None, "min": None, "standard": None},
        }
        # 开始捕捉视频帧并进行校准
        self.capture_video_for_calibration()
    
    def stop_calibration(self):
        # 停止视频捕捉并保存校准数据
        self.is_calibrating = False
        self.save_calibration_data()

    def capture_video_for_calibration(self):
        """捕捉视频帧并计算AUs的最大值、最小值和标准值"""
        # 初始化校准标志
        self.is_calibrating = True
        
        # 初始化视频捕获
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            return
        
        # 初始化校准阶段
        self.calibration_phases = [
            {"name": "Furrow Brow", "duration": 3, "target": "BrowFurrow"},
            {"name": "Raise Eyelids", "duration": 3, "target": "UpperLidRaiser"},
            {"name": "Squint Eyes", "duration": 3, "target": "EyeSquint"},
            {"name": "Smile", "duration": 3, "target": "Smile"},
            {"name": "Part Lips", "duration": 3, "target": "LipsPart"},
            {"name": "Drop Jaw", "duration": 3, "target": "JawDrop"},
            {"name": "Turn Head", "duration": 3, "target": "head_turn"},
            {"name": "Face Forward", "duration": 3, "target": "head_forward"},
            {"name": "Show Facial Activity", "duration": 3, "target": "face_active"}
        ]
        
        # 当前校准阶段索引
        self.current_phase_index = 0
        # 当前阶段剩余时间
        self.phase_time_remaining = self.calibration_phases[0]["duration"]
        # 上次更新时间
        self.last_update_time = time.time()
        
        # 收集的数据
        self.collected_data = {key: [] for key in self.calibration_data.keys()}
        
        # 创建校准指示UI
        self.calibration_info = tk.Label(self.tab2, text=f"请准备{self.calibration_phases[0]['name']}的表情", 
                                        font=("Helvetica", 16))
        self.calibration_info.pack(pady=5)
        
        # 添加当前状态信息标签
        self.status_label = tk.Label(self.tab2, text="准备中，请点击开始采集按钮", 
                                    font=("Helvetica", 12), fg="blue")
        self.status_label.pack(pady=5)
        
        # 添加进度条
        self.progress_bar = ttk.Progressbar(self.tab2, length=500, mode='determinate')
        self.progress_bar.pack(pady=5)
        
        # 更新进度条最大值
        total_time = sum(phase["duration"] for phase in self.calibration_phases)
        self.progress_bar["maximum"] = total_time
        self.progress_bar["value"] = 0
        
        # 添加控制按钮
        self.control_frame = ttk.Frame(self.tab2)
        self.control_frame.pack(pady=10)
        
        self.collect_button = ttk.Button(self.control_frame, text="开始采集", 
                                        command=self.start_collecting)
        self.collect_button.pack(side="left", padx=10)
        
        self.next_button = ttk.Button(self.control_frame, text="下一步", 
                                    command=self.goto_next_calibration_step, state="disabled")
        self.next_button.pack(side="left", padx=10)
        
        # 采集状态变量
        self.is_collecting_samples = False
        self.required_samples = 30  # 每个表情需要的样本数
        
        # 启动校准循环
        self.update_calibration_frame()

    def start_collecting(self):
        """开始采集当前表情的样本"""
        self.is_collecting_samples = True
        self.collect_button.config(state="disabled")
        self.status_label.config(text=f"正在采集{self.calibration_phases[self.current_phase_index]['name']}表情样本...", 
                                fg="blue")
        self.countdown_seconds = self.calibration_phases[self.current_phase_index]["duration"]
        self.update_countdown()

    def update_countdown(self):
        """更新倒计时"""
        if self.countdown_seconds > 0:
            self.status_label.config(text=f"正在采集样本，请保持表情 {self.countdown_seconds} 秒", fg="blue")
            self.countdown_seconds -= 1
            self.window.after(1000, self.update_countdown)
        else:
            # 采集完成
            self.is_collecting_samples = False
            count = len(self.collected_data[self.calibration_phases[self.current_phase_index]["target"]])
            if count >= self.required_samples:
                self.status_label.config(text=f"采集完成！获取了 {count} 个样本。请点击下一步按钮继续。", fg="green")
                self.next_button.config(state="normal")
                # 启动按钮闪烁效果
                self.flash_next_button()
            else:
                self.status_label.config(text=f"样本不足({count}/{self.required_samples})，请重新采集", fg="red")
                self.collect_button.config(state="normal", text="重新采集")

    def flash_next_button(self):
        """使下一步按钮闪烁以引起注意"""
        if not hasattr(self, 'button_flash_state'):
            self.button_flash_state = False
        
        # 切换状态
        self.button_flash_state = not self.button_flash_state
        
        if self.button_flash_state:
            self.next_button.config(style="Accent.TButton")
        else:
            self.next_button.config(style="")
        
        # 如果仍在校准中且不是正在采集，则继续闪烁
        if self.is_calibrating and not self.is_collecting_samples and self.next_button.cget('state') == 'normal':
            self.window.after(500, self.flash_next_button)

    def goto_next_calibration_step(self):
        """进入下一个校准步骤"""
        # 取消按钮闪烁
        if hasattr(self, 'button_flash_state'):
            delattr(self, 'button_flash_state')
        
        # 处理当前步骤收集的数据
        current_phase = self.calibration_phases[self.current_phase_index]
        target_au = current_phase["target"]
        
        # 移至下一步骤
        self.current_phase_index += 1
        
        # 检查是否已完成所有步骤
        if self.current_phase_index >= len(self.calibration_phases):
            # 所有步骤完成，保存数据
            self.process_calibration_data()
            return
        
        # 重置状态为新步骤
        next_phase = self.calibration_phases[self.current_phase_index]
        self.calibration_info.config(text=f"请准备{next_phase['name']}的表情")
        self.status_label.config(text="准备中，请点击开始采集按钮", fg="blue")
        self.next_button.config(state="disabled")
        self.collect_button.config(state="normal", text="开始采集")
        self.phase_time_remaining = next_phase["duration"]
        self.last_update_time = time.time()

    def update_calibration_frame(self):
        """更新校准过程的每一帧"""
        if not self.is_calibrating:
            # 清理UI元素
            if hasattr(self, 'calibration_info'):
                self.calibration_info.destroy()
            if hasattr(self, 'status_label'):
                self.status_label.destroy()
            if hasattr(self, 'progress_bar'):
                self.progress_bar.destroy()
            if hasattr(self, 'control_frame'):
                self.control_frame.destroy()
            if self.cap is not None:
                self.cap.release()
            return
        
        # 读取一帧
        ret, frame = self.cap.read()
        if not ret:
            self.is_calibrating = False
            messagebox.showerror("错误", "无法读取视频帧")
            return
        
        # 检测AUs
        aus_values = self.detect_aus(frame)
    
        # 保存用于绘制的landmarks
        landmarks = None
        if hasattr(self.emotion_recognition, 'last_landmarks') and self.emotion_recognition.last_landmarks is not None:
            landmarks = self.emotion_recognition.last_landmarks
        
        # 如果正在采集样本且检测到AUs，则保存数据
        if self.is_collecting_samples and aus_values:
            current_phase = self.calibration_phases[self.current_phase_index]
            target_au = current_phase["target"]
            
            # 如果检测到目标AU，则保存
            if target_au in aus_values:
                self.collected_data[target_au].append(aus_values[target_au])
                
                # 同时保存其他AU的值，以获得更多数据点
                for key in self.collected_data.keys():
                    if key in aus_values and key != target_au:
                        self.collected_data[key].append(aus_values[key])
        
        # 在帧上绘制校准指示
        current_phase = self.calibration_phases[self.current_phase_index]
        cv2.putText(frame, f"Expression: {current_phase['name']}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
        # 绘制面部关键点和特征（如果有检测到人脸）
        if landmarks is not None and len(landmarks) == 68:
            self.draw_facial_landmarks(frame, landmarks)
        
        # 如果检测到当前阶段的目标AU，显示实时值
        if aus_values and current_phase["target"] in aus_values:
            current_value = aus_values[current_phase["target"]]
            cv2.putText(frame, f"Current Value: {current_value:.3f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 如果正在采集样本，显示样本计数
            if self.is_collecting_samples:
                sample_count = len(self.collected_data[current_phase["target"]])
                cv2.putText(frame, f"Samples: {sample_count}/{self.required_samples}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 显示帧
        cv_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(cv_image_rgb)
        tk_image = PIL.ImageTk.PhotoImage(image=pil_image)
        self.emotion_canvas.image = tk_image
        self.emotion_canvas.create_image(320, 240, image=tk_image)
        
        # 更新进度指示器
        completed_time = sum(phase["duration"] for phase in self.calibration_phases[:self.current_phase_index])
        completed_percent = (self.current_phase_index / len(self.calibration_phases)) * 100
        self.progress_bar["value"] = completed_time
        
        # 继续循环
        self.window.after(30, self.update_calibration_frame)

    def process_calibration_data(self):
        """处理收集到的校准数据，计算最大值、最小值和标准值"""
        # 显示处理中消息
        self.status_label.config(text="正在处理校准数据...", fg="blue")
        
        # 处理每个AU的收集数据
        for key, values in self.collected_data.items():
            if values:  # 确保有收集到的数据
                # 过滤掉异常值
                filtered_values = values
                if len(values) > 10:  # 只有在数据量足够的情况下才过滤
                    # 简单过滤：移除最高和最低的5%
                    filtered_values = sorted(values)[int(len(values)*0.05):int(len(values)*0.95)]
                
                if filtered_values:
                    self.calibration_data[key]["max"] = max(filtered_values)
                    self.calibration_data[key]["min"] = min(filtered_values)
                    # 标准值可以取平均值而不是简单的中间值
                    self.calibration_data[key]["standard"] = sum(filtered_values) / len(filtered_values)
        
        # 保存校准数据
        self.save_calibration_data()
        
        # 显示校准结果
        self.show_calibration_results()

    def show_calibration_results(self):
        """显示校准结果摘要"""
        # 清理UI元素
        if hasattr(self, 'calibration_info'):
            self.calibration_info.destroy()
        if hasattr(self, 'status_label'):
            self.status_label.destroy()
        if hasattr(self, 'progress_bar'):
            self.progress_bar.destroy()
        if hasattr(self, 'control_frame'):
            self.control_frame.destroy()
        
        # 创建结果显示框架
        result_frame = ttk.LabelFrame(self.tab2, text="校准已完成")
        result_frame.pack(pady=10, fill="x", padx=10)
        
        # 添加关闭按钮
        close_button = ttk.Button(self.tab2, text="完成", command=self.finish_calibration)
        close_button.pack(pady=10)
        
        # 显示成功消息
        messagebox.showinfo("校准完成", "面部表情校准已完成，数据已保存到calibration_data.json。")

    def finish_calibration(self):
        """完成校准，清理界面"""
        # 清理UI元素
        for widget in self.tab2.winfo_children():
            widget.destroy()
            
        # 重新设置校准标签页
        self.setup_calibration_tab()
        
        # 重置校准标志
        self.is_calibrating = False

    def save_calibration_data(self):
        """保存校准数据到文件或变量中"""
        # 处理可能的None值
        for au, data in self.calibration_data.items():
            if data["max"] is None or data["min"] is None:
                # 设置默认值
                data["max"] = 1.0
                data["min"] = 0.0
                data["standard"] = 0.5
            elif data["standard"] is None:
                # 计算标准值
                data["standard"] = (data["max"] + data["min"]) / 2
        
        # 保存数据到文件
        try:
            with open("calibration_data.json", 'w') as f:
                json.dump(self.calibration_data, f, indent=4)
            print("校准数据已保存到calibration_data.json")
        except Exception as e:
            messagebox.showerror("保存错误", f"保存校准数据时出错: {str(e)}")

    def save_to_file(self, data, filename):
        # 保存数据到文件
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)    

    def detect_aus(self, frame):
        """检测面部动作单元(AUs)并返回计算的值"""
        # 创建结果字典存储检测到的AU值
        aus_values = {}
        
        # 使用emotion_recognition获取面部特征点
        landmarks_result = self.emotion_recognition.get_landmarks_dlib(frame)
        if landmarks_result is None:
            return None  # 没有检测到人脸
        
        # 根据emotion_recognition.py的实现，正确解析landmarks结果
        if isinstance(landmarks_result, tuple) and len(landmarks_result) == 2:
            landmarks, face_rect = landmarks_result
        else:
            landmarks = landmarks_result
        
        # 确保landmarks是numpy数组格式
        landmarks = np.array(landmarks)
        
        # 估计头部姿态
        head_pose = self.emotion_recognition.estimate_head_pose(landmarks, frame)
        if head_pose:
            pitch, yaw, roll = head_pose[:3]
        else:
            pitch, yaw, roll = 0, 0, 0
        
        # 计算眼睛开合度 - 修复传递给calculate_ear的参数
        # 左眼landmarks通常是36-41，右眼是42-47
        left_eye_points = landmarks[36:42] if len(landmarks) > 41 else None
        right_eye_points = landmarks[42:48] if len(landmarks) > 47 else None
        
        if left_eye_points is not None and right_eye_points is not None:
            left_ear = self.emotion_recognition.calculate_ear(left_eye_points, "left")
            right_ear = self.emotion_recognition.calculate_ear(right_eye_points, "right")
            avg_ear = (left_ear + right_ear) / 2 if left_ear and right_ear else 0
        else:
            avg_ear = 0
        
        # 计算动作变化
        micro_expression = False
        movement_magnitude = 0
        if self.emotion_recognition.last_landmarks is not None:
            # 计算面部关键点变化
            movement = np.linalg.norm(landmarks - self.emotion_recognition.last_landmarks, axis=1)
            movement_magnitude = np.mean(movement)
            
            # 微表情判断
            mouth_movement = np.mean(movement[48:68]) if len(movement) > 67 else 0  # 嘴部区域移动
            eye_movement = np.mean(movement[36:48]) if len(movement) > 47 else 0    # 眼睛区域移动
            brow_movement = np.mean(movement[17:27]) if len(movement) > 26 else 0   # 眉毛区域移动
            
            # 微表情是局部的小幅度变化
            if (2.8 < mouth_movement < 6 or 
                2.8 < eye_movement < 4 or 
                2.8 < brow_movement < 6):
                micro_expression = True
        
        self.emotion_recognition.last_landmarks = landmarks.copy()
        
        # 检测头部稳定性
        head_stable = True
        if hasattr(self.emotion_recognition, 'head_pose_history') and len(self.emotion_recognition.head_pose_history) > 5:
            recent_yaws = [pose[1] for pose in list(self.emotion_recognition.head_pose_history)[-5:]]
            recent_pitches = [pose[0] for pose in list(self.emotion_recognition.head_pose_history)[-5:]]
            yaw_variation = np.std(recent_yaws)
            pitch_variation = np.std(recent_pitches)
            head_stable = yaw_variation < 12 or pitch_variation < 12
        
        # 计算各种AU值
        # 头部方向
        aus_values['head_forward'] = max(0, 1 - abs(yaw / 15.0))  # 前视程度
        aus_values['head_turn'] = min(1.0, abs(yaw) / 30.0)  # 转头程度
        
        # 面部运动
        aus_values['frequent_movement'] = min(1.0, movement_magnitude / 5.0)
        aus_values['face_active'] = min(1.0, movement_magnitude / 10.0)
        aus_values['no_micro_expression'] = 1 - int(micro_expression)
        
        # 眉毛特征
        try:
            if len(landmarks) > 27:
                brow_height = (landmarks[21][1] + landmarks[22][1]) / 2 - landmarks[27][1]
                left_brow_height = landmarks[21][1] - landmarks[27][1]
                right_brow_height = landmarks[22][1] - landmarks[27][1]
                
                aus_values["BrowFurrow"] = min(1.0, max(0, -brow_height / 20.0))
                aus_values["BrowFurrowAsymmetry"] = min(1.0, abs(left_brow_height - right_brow_height) / 6.0)
        except:
            pass
        
        # 眼睛特征
        try:
            if len(landmarks) > 46:
                left_eye_height = np.linalg.norm(landmarks[37] - landmarks[41])
                right_eye_height = np.linalg.norm(landmarks[44] - landmarks[46])
                eye_height_avg = (left_eye_height + right_eye_height) / 2
                
                left_eye_width = np.linalg.norm(landmarks[36] - landmarks[39])
                right_eye_width = np.linalg.norm(landmarks[42] - landmarks[45])
                eye_width_avg = (left_eye_width + right_eye_width) / 2
                
                left_eye_ratio = left_eye_height / left_eye_width if left_eye_width > 0 else 0
                right_eye_ratio = right_eye_height / right_eye_width if right_eye_width > 0 else 0
                eye_ratio_avg = (left_eye_ratio + right_eye_ratio) / 2
                
                aus_values["UpperLidRaiser"] = min(1.0, eye_height_avg / 18.0)
                aus_values["EyeSquint"] = 1.0 - min(1.0, eye_ratio_avg / 0.5)
        except:
            pass
        
        # 嘴部特征
        try:
            if len(landmarks) > 66:
                mouth_corner_height = (landmarks[54][1] + landmarks[48][1]) / 2
                mouth_center_height = landmarks[57][1]
                mouth_open_height = np.linalg.norm(landmarks[62] - landmarks[66])
                mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])
                
                # 微笑程度 - 嘴角相对于嘴中心的高度
                smile_value = (mouth_center_height - mouth_corner_height) / 15.0
                aus_values["Smile"] = min(1.0, max(0, smile_value))
                
                # 张嘴程度
                aus_values["LipsPart"] = min(1.0, mouth_open_height / 20.0)
                
                # 下颌下降
                aus_values["JawDrop"] = min(1.0, mouth_open_height / 30.0)
        except:
            pass
        
        return aus_values
               
            
    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.display_image(self.current_image, self.canvas)
    
    def display_image(self, cv_image, canvas):
        # 转换OpenCV图像为Tkinter可显示的格式
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(cv_image_rgb)
        
        # 调整图像大小以适应canvas
        canvas_width = canvas.winfo_width() or 400
        canvas_height = canvas.winfo_height() or 300
        
        pil_image = self.resize_image(pil_image, canvas_width, canvas_height)
        
        # 创建PhotoImage对象
        tk_image = PIL.ImageTk.PhotoImage(image=pil_image)
        
        # 保存引用以防止图像被垃圾回收
        canvas.image = tk_image
        
        # 显示图像
        canvas.create_image(canvas_width//2, canvas_height//2, image=tk_image)
    
    def resize_image(self, pil_image, width, height):
        # 计算调整比例
        w, h = pil_image.size
        aspect_ratio = min(width/w, height/h)
        new_size = (int(w * aspect_ratio), int(h * aspect_ratio))
        
        return pil_image.resize(new_size, PIL.Image.LANCZOS)
    
    def register_face(self):
        name = self.face_name_var.get()
        if hasattr(self, 'current_image') and name:
            result_image, message = self.face_recognition.register_face(self.current_image, name)
            self.display_image(result_image, self.canvas)
            messagebox.showinfo("注册结果", message)
        else:
            messagebox.showerror("错误", "请选择图像并输入名称")
    
    def recognize_face(self):
        if hasattr(self, 'current_image'):
            result_image, message = self.face_recognition.recognize_face(self.current_image)
            self.display_image(result_image, self.canvas)
            self.recognition_result_var.set(message)
        else:
            messagebox.showerror("错误", "请先选择图像")
    
    def list_faces(self):
        faces_text = self.face_recognition.list_registered_faces()
        self.faces_list.delete(1.0, tk.END)
        self.faces_list.insert(tk.END, faces_text)
    
    def delete_face(self):
        name = self.delete_name_var.get()
        if name:
            result = self.face_recognition.delete_face(name)
            messagebox.showinfo("删除结果", result)
            self.list_faces()  # 刷新列表
        else:
            messagebox.showerror("错误", "请输入要删除的人脸名称")
    
    def start_capture(self, target_canvas, process_func):
        if self.is_capturing:
            self.stop_capture()
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            return
        
        self.is_capturing = True
        self.target_canvas = target_canvas
        self.process_func = process_func
        
        # 在新线程中启动视频捕获
        threading.Thread(target=self.update_frame, daemon=True).start()
    
    def update_frame(self):
        while self.is_capturing:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 处理帧
            processed_frame = self.process_func(frame)
            
            # 在主线程中更新UI
            self.window.after(1, lambda: self.display_image(processed_frame, self.target_canvas))
            
            # # 控制帧率
            # time.sleep(0.03)  # 约30fps
    
    def start_live_recognition(self):
        self.start_capture(self.video_canvas, self.face_recognition.recognize_face_stream)
    
    def start_emotion_detection(self):
        self.start_capture(self.emotion_canvas, self.emotion_recognition.detect_expressions)
    
    def stop_capture(self):
        self.is_capturing = False
        if self.cap is not None:
            self.cap.release()
    
    def on_closing(self):
        self.stop_capture()
        self.window.destroy()

    def setup_classroom_monitor_tab(self):
        """设置课堂状态监测标签页"""
        # 创建课堂监测对象
        self.classroom_monitor = ClassroomMonitor()
        
        # 创建分割面板
        paned_window = ttk.PanedWindow(self.tab3, orient=tk.HORIZONTAL)
        paned_window.pack(fill="both", expand=True)
        
        # 左侧面板 - 视频显示
        left_frame = ttk.Frame(paned_window)
        paned_window.add(left_frame, weight=1)
        
        # 视频显示区域
        self.monitor_canvas = tk.Canvas(left_frame, width=640, height=480)
        self.monitor_canvas.pack(pady=10)
        
        # 控制按钮框架
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(pady=5)
        
        # 开始监测按钮
        self.start_monitor_btn = ttk.Button(
            control_frame, 
            text="开始课堂监测", 
            command=self.start_classroom_monitoring
        )
        self.start_monitor_btn.pack(side="left", padx=5)
        
        # 停止按钮
        self.stop_monitor_btn = ttk.Button(
            control_frame, 
            text="停止监测", 
            command=self.stop_capture
        )
        self.stop_monitor_btn.pack(side="left", padx=5)
        
        # 生成理解度趋势图按钮
        self.generate_trend_btn = ttk.Button(
            control_frame, 
            text="生成理解度报告", 
            command=self.generate_understanding_trends
        )
        self.generate_trend_btn.pack(side="left", padx=5)
        
        # 右侧面板 - 实时趋势图
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight=1)
        
        # 趋势图标题
        ttk.Label(right_frame, text="Real-time Understanding Trends", 
                font=("Arial", 12, "bold")).pack(pady=5)
        
        # 创建matplotlib图形并嵌入tkinter
        self.trend_figure = self.classroom_monitor.setup_realtime_chart()
        self.trend_canvas = FigureCanvasTkAgg(self.trend_figure, right_frame)
        self.trend_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # 添加说明文本
        info_frame = ttk.LabelFrame(left_frame, text="Classroom Monitoring Guide")
        info_frame.pack(pady=10, fill="x", padx=10)
        
        info_text = """
        Real-time Classroom Monitoring System:
        
        1. Click "Start Monitoring" to collect student data
        2. The trend chart will update automatically
        3. Different colors represent different students
        4. Click "Generate Report" to save trend images
        """
        ttk.Label(info_frame, text=info_text, justify="left").pack(pady=5)

    def start_classroom_monitoring(self):
        """开始课堂监测并启动趋势图更新"""
        # 启动视频捕获
        self.start_capture(self.monitor_canvas, self.classroom_monitor.process_frame)
        
        # 启动趋势图更新定时器
        self.update_trend_chart()

    def update_trend_chart(self):
        """定时更新趋势图"""
        if self.is_capturing:
            # 更新趋势图
            self.classroom_monitor.update_realtime_chart()
            
            # 每秒更新一次
            self.window.after(1000, self.update_trend_chart)

    def generate_understanding_trends(self):
        """生成理解度趋势图报告"""
        # 选择保存目录
        output_dir = filedialog.askdirectory(title="Select Save Location")
        if not output_dir:
            return  # 用户取消
        
        # 显示进度对话框
        progress = tk.Toplevel(self.window)
        progress.title("Generating Reports")
        progress.geometry("300x100")
        progress.transient(self.window)
        progress.grab_set()
        
        ttk.Label(progress, text="Generating understanding trend reports...").pack(pady=20)
        progress.update()
        
        # 生成趋势图
        try:
            report_count = self.classroom_monitor.generate_understanding_reports(output_dir)
            progress.destroy()
            
            if report_count > 0:
                if messagebox.askyesno("Success", 
                                f"Generated {report_count} student trend reports.\n\nOpen folder now?"):
                    if sys.platform == 'win32':
                        os.startfile(output_dir)
                    elif sys.platform == 'darwin':  # macOS
                        os.system(f'open "{output_dir}"')
                    else:  # Linux
                        os.system(f'xdg-open "{output_dir}"')
            else:
                messagebox.showwarning("No Data", 
                                "Not enough data to generate trend reports.\nEnsure at least 5 data points per student.")
        except Exception as e:
            progress.destroy()
            messagebox.showerror("Error", f"Error generating reports:\n{str(e)}")

    def start_expression_calibration(self):
        """启动表情校准流程"""
        name = self.face_name_var.get()
        if not name:
            messagebox.showerror("Error", "Please enter a face name/ID first")
            return
        
        # 检查ID是否已注册
        if name not in self.face_recognition.face_database:
            if not messagebox.askyesno("Tip", f"'{name}' is not registered. Register the face first?"):
                return
            
            # 先注册人脸
            if not hasattr(self, 'current_image'):
                messagebox.showerror("Error", "Please select an image or take a photo first")
                return
            
            result_image, message = self.face_recognition.register_face(self.current_image, name)
            if "Success" not in message:
                messagebox.showerror("Registration Failed", message)
                return
        
        # 显示校准说明
        instruction = """
        Expression Calibration Instructions:
        
        1. You will be guided through 9 different facial expressions
        2. For each expression, follow the on-screen instructions
        3. Hold each expression steady until samples are collected 
        4. Click the 'NEXT STEP' button when it becomes active
        5. Complete all steps for the best calibration results
        
        This process takes about 2-3 minutes.
        """
        
        if messagebox.askokcancel("Start Calibration", instruction):
            # 创建校准对象并启动校准
            calibration = ExpressionCalibration(self.face_recognition, self.emotion_recognition)
            calibration.start_calibration(name)

    def start_calibration_window(self, user_id):
        """创建并显示校准窗口"""
        # 创建校准窗口
        calibration_window = tk.Toplevel(self.window)
        calibration_window.title(f"表情校准 - {user_id}")
        calibration_window.geometry("800x600")
        
        # 初始化校准过程
        self.calibration_module.start_calibration(user_id)
        
        # 校准视频显示区域
        calibration_canvas = tk.Canvas(calibration_window, width=640, height=480)
        calibration_canvas.pack(pady=10)
        
        # 校准提示区域
        prompt_var = tk.StringVar(value=self.calibration_module.get_current_prompt())
        prompt_label = ttk.Label(
            calibration_window, 
            textvariable=prompt_var,
            font=("Arial", 14, "bold")
        )
        prompt_label.pack(pady=10)
        
        # 倒计时区域
        countdown_var = tk.StringVar(value="准备开始...")
        countdown_label = ttk.Label(
            calibration_window, 
            textvariable=countdown_var,
            font=("Arial", 20)
        )
        countdown_label.pack(pady=10)
        
        # 开始校准按钮
        start_btn = ttk.Button(
            calibration_window, 
            text="开始采集", 
            command=lambda: self.run_calibration_step(
                calibration_window, prompt_var, countdown_var, start_btn
            )
        )
        start_btn.pack(pady=10)
        
        # 启动摄像头
        self.start_capture(calibration_canvas, self.calibration_module.process_frame_for_calibration)

    def run_calibration_step(self, window, prompt_var, countdown_var, button):
        """运行当前校准步骤"""
        button.config(state="disabled")
        
        # 倒计时
        def countdown(count):
            if count > 0:
                countdown_var.set(f"采集中... {count}")
                window.after(1000, lambda: countdown(count-1))
            else:
                # 进入下一步，并获取结果
                self.calibration_module.next_step()
                done = self.calibration_module.current_step_index >= len(self.calibration_module.calibration_steps)
                message = self.calibration_module.get_current_prompt()
                prompt_var.set(message)
                
                if done:
                    # 校准完成
                    countdown_var.set("校准完成!")
                    messagebox.showinfo("校准完成", "表情校准已完成，设置已保存。")
                    self.stop_capture()
                    window.destroy()
                else:
                    # 进入下一步
                    countdown_var.set("准备下一步...")
                    button.config(state="normal", text="继续下一步")
        
        # 开始倒计时
        countdown(5)  # 5秒采集时间

    def draw_facial_landmarks(self, frame, landmarks):
        """绘制面部关键点和重要特征"""
        # 绘制所有关键点
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # 绘制眼睛轮廓
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        cv2.polylines(frame, [np.array(left_eye, dtype=np.int32)], True, (0, 255, 255), 1)
        cv2.polylines(frame, [np.array(right_eye, dtype=np.int32)], True, (0, 255, 255), 1)
        
        # 绘制嘴部轮廓
        mouth = landmarks[48:60]
        cv2.polylines(frame, [np.array(mouth, dtype=np.int32)], True, (0, 255, 255), 1)
        
        # 绘制眉毛
        left_eyebrow = landmarks[17:22]
        right_eyebrow = landmarks[22:27]
        cv2.polylines(frame, [np.array(left_eyebrow, dtype=np.int32)], False, (0, 255, 255), 1)
        cv2.polylines(frame, [np.array(right_eyebrow, dtype=np.int32)], False, (0, 255, 255), 1)
        
        # 绘制下巴轮廓
        jaw = landmarks[0:17]
        cv2.polylines(frame, [np.array(jaw, dtype=np.int32)], False, (0, 255, 255), 1)
        
        # 绘制内嘴唇轮廓
        inner_mouth = landmarks[60:68]
        cv2.polylines(frame, [np.array(inner_mouth, dtype=np.int32)], True, (0, 255, 255), 1)
        
        # 绘制鼻子
        nose_bridge = landmarks[27:31]
        nose_tip = landmarks[31:36]
        cv2.polylines(frame, [np.array(nose_bridge, dtype=np.int32)], False, (0, 255, 255), 1)
        cv2.polylines(frame, [np.array(nose_tip, dtype=np.int32)], False, (0, 255, 255), 1)
        
        return frame




def create_ui():
    """创建并启动人脸分析应用的用户界面"""
    root = tk.Tk()
    root.title("人脸分析系统")
    
    # 设置窗口大小和位置
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = 900
    window_height = 700
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    # 确保目录存在
    os.makedirs("model", exist_ok=True) 
    os.makedirs("data", exist_ok=True)
    
    # 创建应用实例
    try:
        app = FaceAnalysisApp(root, "人脸分析系统")
        return app
    except Exception as e:
        print(f"启动应用时发生错误: {str(e)}")
        messagebox.showerror("启动错误", f"启动应用时发生错误: {str(e)}")
        root.destroy()
        return None

# 如果直接运行app.py，也启动UI
if __name__ == "__main__":
    create_ui()