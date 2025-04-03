import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import os
import json
import time
from collections import deque
import PIL.Image, PIL.ImageTk

class ExpressionCalibration:
    def __init__(self, face_recognition, emotion_recognition):
        # 关联识别模块
        self.face_recognition = face_recognition
        self.emotion_recognition = emotion_recognition
        
        # 校准数据存储
        self.calibration_data = {
            "neutral": {},       # Neutral expression - baseline
            "eye_open": {},      # Eyes fully open 
            "eye_closed": {},    # Eyes closed
            "smile": {},         # Smile
            "frown": {},         # Frown
            "confused": {},      # Confused expression
            "tired": {},         # Tired expression
            "focused": {},       # Focused state
            "distracted": {}     # Distracted state
        }
        
        # 各表情状态的指导提示
        self.calibration_prompts = {
            "neutral": "Keep a neutral expression, face the camera",
            "eye_open": "Open your eyes fully and look at the camera",
            "eye_closed": "Close your eyes completely for 5 seconds",
            "smile": "Smile naturally as if something pleased you",
            "frown": "Frown your eyebrows as if concentrating",
            "confused": "Show a confused expression (raise one eyebrow)",
            "tired": "Show a tired expression (half-close eyes)",
            "focused": "Show a focused reading expression (slight frown)",
            "distracted": "Look around, as if distracted by something"
        }
        
        # 校准步骤
        self.calibration_steps = list(self.calibration_prompts.keys())
        self.current_step_index = 0
        
        # 每个步骤的采样数据
        self.step_samples = []
        self.required_samples = 5
        
        # 用户ID
        self.user_id = "default"
        
        # UI相关变量
        self.calibration_window = None
        self.video_canvas = None
        self.cap = None
        self.is_running = False
        
        # 确保目录存在
        os.makedirs("data/calibration", exist_ok=True)
    
    def start_calibration(self, user_id=None):
        """开始校准过程"""
        if user_id:
            self.user_id = user_id
        
        # 重置校准状态
        self.current_step_index = 0
        self.step_samples = []
        
        # 创建并显示窗口
        self.create_calibration_window()
        
        return True
    
    def create_calibration_window(self):
        """创建校准窗口和UI元素"""
        # 创建主窗口
        self.calibration_window = tk.Toplevel()
        self.calibration_window.title(f"Expression Calibration - {self.user_id}")
        self.calibration_window.geometry("800x700")  # 增加高度
        self.calibration_window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 主框架
        main_frame = ttk.Frame(self.calibration_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 提示标签 - 使用更大的字体和明显的颜色
        self.prompt_frame = ttk.Frame(main_frame, padding=10)
        self.prompt_frame.pack(fill="x", pady=10)
        
        self.prompt_label = ttk.Label(
            self.prompt_frame, 
            text=self.get_current_prompt(), 
            font=("Arial", 16, "bold"),
            wraplength=700,
            anchor="center",
            justify="center"
        )
        self.prompt_label.pack(pady=10)
        
        # 参考图片展示区 (未来可添加表情参考图)
        self.reference_frame = ttk.Frame(main_frame)
        self.reference_frame.pack(pady=5)
        
        # 视频显示区域
        video_frame = ttk.Frame(main_frame)
        video_frame.pack(pady=10)
        
        # 视频画布
        self.video_canvas = tk.Canvas(video_frame, width=640, height=360, bg="black")
        self.video_canvas.pack()
        
        # 指标显示区
        self.metrics_frame = ttk.LabelFrame(main_frame, text="Facial Metrics")
        self.metrics_frame.pack(fill="x", padx=5, pady=5)
        
        metrics_grid = ttk.Frame(self.metrics_frame)
        metrics_grid.pack(padx=10, pady=5)
        
        # 添加显示当前测量的指标
        self.metric_labels = {}
        metrics = ["Eye Ratio", "Mouth Open", "Smile", "Eyebrow Height", "Asymmetry"]
        for i, metric in enumerate(metrics):
            ttk.Label(metrics_grid, text=f"{metric}:").grid(row=i//3, column=(i%3)*2, sticky="e", padx=5, pady=2)
            label = ttk.Label(metrics_grid, text="0.00")
            label.grid(row=i//3, column=(i%3)*2+1, sticky="w", padx=5, pady=2)
            self.metric_labels[metric] = label
        
        # 进度指示器
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(pady=10, fill="x")
        
        self.progress_label = ttk.Label(
            progress_frame, 
            text=f"Step {self.current_step_index+1}/{len(self.calibration_steps)}",
            font=("Arial", 12)
        )
        self.progress_label.pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", 
                                          length=700, mode="determinate")
        self.progress_bar.pack(pady=5)
        self.progress_bar["maximum"] = len(self.calibration_steps)
        self.progress_bar["value"] = self.current_step_index
        
        # 样本计数
        self.sample_label = ttk.Label(
            progress_frame, 
            text=f"Samples: 0/{self.required_samples}",
            font=("Arial", 12)
        )
        self.sample_label.pack(pady=5)
    
        # 添加是否在采集状态的标志
        self.is_collecting = False
        
        # 直接启动视频捕获，但不开始采集
        self.start_video_capture()
        
        # 按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        # 添加开始采集按钮
        self.start_collection_btn = ttk.Button(
            button_frame, 
            text="开始采集", 
            command=self.start_current_collection,
            width=20
        )
        self.start_collection_btn.pack(side=tk.LEFT, padx=20)
        
        # 下一步按钮
        self.next_btn = ttk.Button(
            button_frame, 
            text="下一步", 
            command=self.next_step,
            width=20
        )
        self.next_btn.pack(side=tk.LEFT, padx=20)
        self.next_btn.config(state=tk.DISABLED)  # 初始禁用
        
        # 取消按钮
        cancel_btn = ttk.Button(
            button_frame, 
            text="取消", 
            command=self.on_closing,
            width=15
        )
        cancel_btn.pack(side=tk.LEFT, padx=20)
    
        # 说明文本
        instruction_frame = ttk.LabelFrame(main_frame, text="Instructions")
        instruction_frame.pack(fill="x", padx=10, pady=10)
        
        instructions = """
        1. 每个表情，点击"开始采集"并保持姿势直到收集完成
        2. 采集完成后，可以点击"重新采集"尝试更好的样本
        3. 对当前采集结果满意后，点击"下一步"继续
        4. 完成所有步骤以获得准确的校准结果
        """
        ttk.Label(instruction_frame, text=instructions, justify="left").pack(pady=5)
    
    
    def start_current_collection(self):
        """开始当前表情的数据采集"""
        self.is_collecting = True
        self.step_samples = []  # 清空先前可能的样本
        
        # 更新按钮状态
        self.start_collection_btn.config(state=tk.DISABLED)
        self.start_collection_btn.config(text="正在采集...")
        
        # 更新文本提示
        self.sample_label.config(
            text="保持表情，正在采集样本...",
            foreground="blue"
        )

    def update_countdown(self):
        """更新倒计时显示"""
        if self.countdown_seconds > 0:
            self.sample_label.config(
                text=f"保持表情，正在采集样本... {self.countdown_seconds}秒",
                foreground="blue"
            )
            self.countdown_seconds -= 1
            self.calibration_window.after(1000, self.update_countdown)
        else:
            # 只有当没有运行视频捕获时才启动
            if not self.is_running and self.cap is None:
                self.start_video_capture()
    
    def start_video_capture(self):
        """开始视频捕获和处理"""
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        self.update_video()
    
    def update_video(self):
        """更新视频流并处理帧"""
        if not self.is_running:
            return
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                return
                    
            # 处理帧进行校准
            processed_frame = self.process_frame_for_calibration(frame)
            
            # 转换为tkinter格式
            if not hasattr(self, 'photo'):
                # 第一次初始化
                img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = PIL.Image.fromarray(img)
                self.photo = PIL.ImageTk.PhotoImage(image=img)
            else:
                # 重用现有对象
                img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = PIL.Image.fromarray(img)
                self.photo.paste(img)
            
            # 更新画布
            self.video_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            
            # 只有在采集状态下才收集样本
            if self.is_collecting:
                # 更新样本计数
                self.sample_label.config(text=f"采集中: {len(self.step_samples)}/{self.required_samples}")
                
                # 样本足够时启用下一步按钮并更改开始按钮为重新采集
                if len(self.step_samples) >= self.required_samples:
                    self.is_collecting = False
                    self.next_btn.config(state=tk.NORMAL, text="下一步 →")
                    
                    # 将开始采集按钮改为重新采集按钮
                    self.start_collection_btn.config(
                        state=tk.NORMAL, 
                        text="重新采集", 
                        command=self.restart_collection
                    )
                    
                    # 简化闪烁效果
                    if not hasattr(self, 'flash_timer'):
                        self.flash_timer = self.calibration_window.after(500, self.flash_next_button)
                    
                    # 提示文本也高亮显示
                    self.sample_label.config(
                        text="✓ 采集完成 - 点击下一步按钮继续或重新采集",
                        foreground="green"
                    )
        except Exception as e:
            print(f"Error in video update: {e}")
        
        # 继续更新
        if self.calibration_window:
            self.video_timer = self.calibration_window.after(30, self.update_video)

    # 添加重新采集方法
    def restart_collection(self):
        """重新开始当前表情的采集"""
        # 清空当前样本
        self.step_samples = []
        
        # 重置采集状态
        self.is_collecting = True
        
        # 更新按钮状态
        self.start_collection_btn.config(
            state=tk.DISABLED,
            text="正在采集...",
            command=self.start_current_collection  # 恢复原来的命令
        )
        
        # 禁用下一步按钮
        self.next_btn.config(state=tk.DISABLED, text="下一步")
        
        # 取消闪烁效果
        if hasattr(self, 'flash_timer'):
            self.calibration_window.after_cancel(self.flash_timer)
            delattr(self, 'flash_timer')
        
        # 更新文本提示
        self.sample_label.config(
            text="重新采集样本中...",
            foreground="blue"
        )

    # 添加新方法来处理按钮闪烁
    def flash_next_button(self):
        """处理下一步按钮闪烁"""
        if not hasattr(self, 'button_flash_state'):
            self.button_flash_state = False
        
        self.button_flash_state = not self.button_flash_state
        
        if self.button_flash_state:
            self.next_btn.config(style="Accent.TButton")
        else:
            self.next_btn.config(style="")
        
        # 继续闪烁直到下一步按钮被点击
        if self.calibration_window and self.is_running:
            self.flash_timer = self.calibration_window.after(500, self.flash_next_button)
    
    def get_current_prompt(self):
        """获取当前校准步骤的提示"""
        if self.current_step_index < len(self.calibration_steps):
            current_step = self.calibration_steps[self.current_step_index]
            return self.calibration_prompts[current_step]
        return "Calibration complete"
    
    def process_frame_for_calibration(self, frame):
        """处理校准帧"""
        result_frame = frame.copy()
        current_step = self.calibration_steps[self.current_step_index]
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_recognition.detector(gray, 0)
            
            if len(faces) == 0:
                cv2.putText(result_frame, "No face detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return result_frame
            
            # 获取人脸和关键点
            face = faces[0]
            shape = self.face_recognition.shape_predictor(gray, face)
            landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            
            # 提取面部指标
            metrics = self.extract_face_metrics(landmarks, frame)
            
            if metrics:
                # 只有在采集状态下才收集样本
                if self.is_collecting:
                    self.step_samples.append(metrics)
                    if len(self.step_samples) > self.required_samples:
                        self.step_samples = self.step_samples[-self.required_samples:]
                
                # 更新指标显示
                self.update_metric_display(metrics)
                
                # 在视频上显示信息
                cv2.putText(result_frame, f"Step: {current_step}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                if self.is_collecting:
                    cv2.putText(result_frame, f"Collecting: {len(self.step_samples)}/{self.required_samples}", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # 样本足够时给用户明显提示
                    if len(self.step_samples) >= self.required_samples:
                        cv2.putText(
                            result_frame, 
                            "CLICK 'NEXT STEP' BUTTON TO CONTINUE",
                            (frame.shape[1]//2 - 250, frame.shape[0] - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, 
                            (0, 255, 255), 
                            2
                        )
                else:
                    cv2.putText(result_frame, "Click 'START COLLECTION' when ready", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # 绘制关键面部标记点
                self.draw_facial_landmarks(result_frame, landmarks)
                
            else:
                cv2.putText(result_frame, "Cannot extract facial features", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        except Exception as e:
            print(f"Error in calibration frame processing: {e}")
            cv2.putText(result_frame, f"Error: {str(e)[:30]}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return result_frame
    
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
    
    def update_metric_display(self, metrics):
        """更新UI上的指标显示"""
        if 'ear' in metrics:
            self.metric_labels["Eye Ratio"].config(text=f"{metrics['ear']:.3f}")
        
        if 'mouth_open' in metrics:
            self.metric_labels["Mouth Open"].config(text=f"{metrics['mouth_open']:.3f}")
        
        if 'mouth_corner_lift' in metrics:
            self.metric_labels["Smile"].config(text=f"{metrics['mouth_corner_lift']:.3f}")
        
        if 'brow_height' in metrics:
            self.metric_labels["Eyebrow Height"].config(text=f"{metrics['brow_height']:.3f}")
        
        if 'brow_asymmetry' in metrics:
            self.metric_labels["Asymmetry"].config(text=f"{metrics['brow_asymmetry']:.3f}")
    
    def extract_face_metrics(self, landmarks, image):
        """提取面部指标"""
        if landmarks is None or len(landmarks) < 68:
            return None
        
        metrics = {}
        
        # 1. 眼睛长宽比(EAR)
        left_eye_ear = self.calculate_ear([landmarks[36], landmarks[37], landmarks[38], 
                                         landmarks[39], landmarks[40], landmarks[41]])
        right_eye_ear = self.calculate_ear([landmarks[42], landmarks[43], landmarks[44], 
                                          landmarks[45], landmarks[46], landmarks[47]])
        metrics['ear'] = (left_eye_ear + right_eye_ear) / 2
        
        # 2. 嘴部开合度
        mouth_height = np.linalg.norm(landmarks[62] - landmarks[66])
        mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])
        metrics['mouth_open'] = mouth_height
        metrics['mouth_ratio'] = mouth_height / (mouth_width + 0.001)
        
        # 3. 嘴角上扬程度 (微笑检测)
        mouth_corner_height = (landmarks[54][1] + landmarks[48][1]) / 2
        mouth_center_height = landmarks[57][1]
        metrics['mouth_corner_lift'] = mouth_center_height - mouth_corner_height
        
        # 4. 眉毛高度 (皱眉检测)
        left_brow_height = landmarks[21][1] - landmarks[27][1]
        right_brow_height = landmarks[22][1] - landmarks[27][1]
        metrics['brow_height'] = (left_brow_height + right_brow_height) / 2
        
        # 5. 眉毛不对称性 (困惑检测)
        metrics['brow_asymmetry'] = abs(left_brow_height - right_brow_height)
        
        # 6. 眼睛眯起程度 (疲劳检测)
        left_eye_height = np.linalg.norm(landmarks[37] - landmarks[41])
        right_eye_height = np.linalg.norm(landmarks[43] - landmarks[47])
        metrics['eye_squint'] = (left_eye_height + right_eye_height) / 2
        
        return metrics
    
    def calculate_ear(self, eye_points):
        """计算眼睛纵横比"""
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def next_step(self):
        """处理下一步按钮点击"""
        # 清理所有计时器
        if hasattr(self, 'flash_timer'):
            self.calibration_window.after_cancel(self.flash_timer)
            delattr(self, 'flash_timer')
        
        if hasattr(self, 'button_flash_state'):
            delattr(self, 'button_flash_state')
        
        # 处理当前步骤
        self.process_step_data()
        
        # 移至下一步
        self.current_step_index += 1
        
        # 检查是否完成校准
        if self.current_step_index >= len(self.calibration_steps):
            self.save_calibration_data()
            # 不再直接关闭，而是显示确认对话框
            self.show_calibration_results()
            return
        
        # 更新UI
        self.prompt_label.config(text=self.get_current_prompt())
        self.progress_bar["value"] = self.current_step_index
        self.progress_label.config(text=f"步骤 {self.current_step_index+1}/{len(self.calibration_steps)}")
        self.sample_label.config(
            text="请准备好表情后点击'开始采集'按钮",
            foreground=""  # 重置文本颜色
        )
        self.next_btn.config(state=tk.DISABLED, text="下一步")
        
        # 重置开始采集按钮的状态和命令
        self.start_collection_btn.config(
            state=tk.NORMAL, 
            text="开始采集", 
            command=self.start_current_collection
        )
        
        # 重置采集状态和样本
        self.is_collecting = False
        self.step_samples = []

    def show_calibration_results(self):
        """显示校准结果供用户确认"""
        # 停止视频处理
        self.is_running = False
        if hasattr(self, 'video_timer'):
            self.calibration_window.after_cancel(self.video_timer)
        
        # 释放摄像头资源
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # 清理当前界面元素
        for widget in self.calibration_window.winfo_children():
            widget.destroy()
        
        # 创建新的结果显示界面
        main_frame = ttk.Frame(self.calibration_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 标题
        ttk.Label(
            main_frame,
            text="校准结果确认",
            font=("Arial", 18, "bold"),
        ).pack(pady=(0, 20))
        
        # 创建滚动区域显示结果
        result_frame = ttk.Frame(main_frame)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        canvas = tk.Canvas(result_frame)
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 格式化显示校准结果
        self.display_calibration_results(scrollable_frame)
        
        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(
            button_frame,
            text="确认并保存",
            command=self.confirm_calibration,
            width=20
        ).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(
            button_frame,
            text="重新校准",
            command=self.restart_calibration,
            width=20
        ).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(
            button_frame,
            text="取消",
            command=self.on_closing,
            width=15
        ).pack(side=tk.LEFT, padx=10)

    def display_calibration_results(self, parent_frame):
        """在结果确认界面显示校准数据"""
        # 显示提取的参数
        params = self.extract_params_from_calibration(self.calibration_data)
        
        ttk.Label(
            parent_frame,
            text="检测阈值参数:",
            font=("Arial", 14, "bold"),
            anchor="w"
        ).pack(fill="x", pady=(10, 5))
        
        # 创建参数表格
        params_frame = ttk.Frame(parent_frame)
        params_frame.pack(fill="x", padx=10, pady=5)
        
        # 格式化阈值参数
        row = 0
        for key, value in params.items():
            ttk.Label(
                params_frame, 
                text=f"{key}:",
                width=25,
                anchor="e"
            ).grid(row=row, column=0, sticky="e", padx=5, pady=2)
            
            if isinstance(value, tuple):
                val_text = f"{value[0]:.3f} - {value[1]:.3f}"
            elif isinstance(value, float):
                val_text = f"{value:.3f}"
            else:
                val_text = str(value)
            
            ttk.Label(
                params_frame, 
                text=val_text,
                width=25,
                anchor="w"
            ).grid(row=row, column=1, sticky="w", padx=5, pady=2)
            
            row += 1
        
        # 显示原始校准数据
        ttk.Label(
            parent_frame,
            text="原始校准数据:",
            font=("Arial", 14, "bold"),
            anchor="w"
        ).pack(fill="x", pady=(20, 5))
        
        # 为每种表情创建一个折叠区域
        for expression, data in self.calibration_data.items():
            if not data:  # 跳过空数据
                continue
                
            # 创建折叠框架
            expander_frame = ttk.LabelFrame(parent_frame, text=f"{expression}")
            expander_frame.pack(fill="x", padx=10, pady=5, ipady=5)
            
            # 显示数据
            data_grid = ttk.Frame(expander_frame)
            data_grid.pack(fill="x", padx=5, pady=5)
            
            # 数据表格
            row = 0
            for key, value in data.items():
                ttk.Label(
                    data_grid, 
                    text=f"{key}:",
                    width=25,
                    anchor="e"
                ).grid(row=row, column=0, sticky="e", padx=5, pady=1)
                
                ttk.Label(
                    data_grid, 
                    text=f"{value:.4f}" if isinstance(value, float) else str(value),
                    width=15,
                    anchor="w"
                ).grid(row=row, column=1, sticky="w", padx=5, pady=1)
                
                row += 1
        
        # 指导信息
        ttk.LabelFrame(
            parent_frame,
            text="说明"
        ).pack(fill="x", padx=10, pady=10)
        
        ttk.Label(
            parent_frame,
            text=(
                "以上数据将用于个性化表情识别阈值，影响疲劳度、专注度和情绪检测的精确性。\n"
                "如果您对校准效果不满意，可以选择重新校准。"
            ),
            wraplength=600,
            justify="left"
        ).pack(fill="x", padx=15, pady=10)

    def confirm_calibration(self):
        """确认校准结果"""
        # 已经在next_step保存了数据，这里只需显示成功消息
        messagebox.showinfo("校准完成", "表情校准结果已保存！系统将使用这些数据进行个性化表情识别。")
        self.on_closing()

    def restart_calibration(self):
        """重新开始校准过程"""
        # 清理当前窗口
        for widget in self.calibration_window.winfo_children():
            widget.destroy()
            
        # 重置校准状态
        self.current_step_index = 0
        self.step_samples = []
        self.calibration_data = {key: {} for key in self.calibration_prompts.keys()}
        
        # 重新创建校准界面
        self.create_calibration_window()

    
    def process_step_data(self):
        """处理当前步骤的数据"""
        if len(self.step_samples) < 1:
            return False
        
        current_step = self.calibration_steps[self.current_step_index]
        
        # 计算指标
        metrics_keys = [
            'ear', 'mouth_open', 'mouth_ratio', 'mouth_corner_lift', 
            'brow_height', 'brow_asymmetry', 'eye_squint'
        ]
        
        # 计算每个指标的平均值和标准差
        for key in metrics_keys:
            values = [sample.get(key, None) for sample in self.step_samples 
                     if key in sample and sample[key] is not None]
            if values:
                self.calibration_data[current_step][f'{key}_avg'] = float(np.mean(values))
                self.calibration_data[current_step][f'{key}_std'] = float(np.std(values))
        
        return True
    
    def save_calibration_data(self):
        """保存校准数据到文件"""
        filename = f"data/calibration/{self.user_id}_calibration.json"
        
        with open(filename, 'w') as f:
            json.dump(self.calibration_data, f, indent=4)
        
        # 更新人脸识别数据库
        if hasattr(self.face_recognition, 'face_database') and self.user_id in self.face_recognition.face_database:
            self.face_recognition.face_database[self.user_id]['calibration_file'] = filename
            self.face_recognition.save_faces()
        
        # 应用到表情识别模块
        try:
            self.apply_calibration_to_emotion_recognition()
            print(f"Calibration for {self.user_id} added to emotion recognition")
        except Exception as e:
            print(f"Warning: Could not apply calibration: {e}")
        
        return True
    
    def apply_calibration_to_emotion_recognition(self):
        """安全地应用校准数据到emotion_recognition"""
        if hasattr(self.emotion_recognition, 'add_calibration_for_id'):
            self.emotion_recognition.add_calibration_for_id(self.user_id, self.calibration_data)
        else:
            if not hasattr(self.emotion_recognition, 'calibration_by_id'):
                setattr(self.emotion_recognition, 'calibration_by_id', {})
                
            params = self.extract_params_from_calibration(self.calibration_data)
            self.emotion_recognition.calibration_by_id[self.user_id] = params
            
            print(f"Added calibration for {self.user_id} manually")
    
    def extract_params_from_calibration(self, calibration_data):
        """从校准数据中提取参数"""
        params = {
            "focused_ear_range": (0.20, 0.30),
            "tired_ear_threshold": 0.17,
            "smile_threshold": -18,
            "brow_height_threshold": 10,
            "brow_asymmetry_threshold": 5,
            "head_stable_threshold": 5,
            "micro_expression_threshold": 3,
            "emotion_decay_rate": 0.75,
            "confidence_threshold": 60
        }
        
        # 眼睛相关参数
        if 'eye_open' in calibration_data and 'eye_closed' in calibration_data:
            if 'ear_avg' in calibration_data['eye_open'] and 'ear_avg' in calibration_data['eye_closed']:
                open_ear = calibration_data['eye_open']['ear_avg']
                closed_ear = calibration_data['eye_closed']['ear_avg']
                
                params["eye_open_threshold"] = open_ear * 0.9
                params["eye_closed_threshold"] = closed_ear * 1.1
                params["tired_ear_threshold"] = closed_ear * 1.2
        
        # 专注状态参数
        if 'focused' in calibration_data and 'ear_avg' in calibration_data['focused']:
            focused_ear = calibration_data['focused']['ear_avg']
            focused_ear_std = calibration_data['focused'].get('ear_std', 0.02)
            params["focused_ear_range"] = (focused_ear - focused_ear_std, focused_ear + focused_ear_std)
        
        # 微笑阈值
        if 'smile' in calibration_data and 'neutral' in calibration_data:
            if 'mouth_corner_lift' in calibration_data['smile'] and 'mouth_corner_lift' in calibration_data['neutral']:
                smile_lift = calibration_data['smile']['mouth_corner_lift']
                neutral_lift = calibration_data['neutral']['mouth_corner_lift']
                params["smile_threshold"] = (smile_lift - neutral_lift) * 0.7 + neutral_lift
        
        # 皱眉阈值
        if 'frown' in calibration_data and 'brow_height_avg' in calibration_data['frown']:
            params["brow_height_threshold"] = calibration_data['frown']['brow_height_avg']
        
        # 眉毛不对称性阈值（困惑表情）
        if 'confused' in calibration_data and 'brow_asymmetry_avg' in calibration_data['confused']:
            params["brow_asymmetry_threshold"] = calibration_data['confused']['brow_asymmetry_avg'] * 0.8
        
        return params
    
    def process_calibration_frame(self, frame):
        """兼容app.py的方法调用"""
        return self.process_frame_for_calibration(frame)

    def complete_current_step(self):
        """兼容app.py的方法调用"""
        self.process_step_data()
        self.current_step_index += 1
        
        if self.current_step_index >= len(self.calibration_steps):
            self.save_calibration_data()
            return True, "Calibration complete"
        
        self.step_samples = []
        return False, self.get_current_prompt()
    
    def on_closing(self):
        """窗口关闭时处理"""
        self.is_running = False
        
        # 取消所有计时器
        if hasattr(self, 'video_timer'):
            self.calibration_window.after_cancel(self.video_timer)
        
        if hasattr(self, 'flash_timer'):
            self.calibration_window.after_cancel(self.flash_timer)
        
        # 释放视频资源
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # 清理图像对象
        if hasattr(self, 'photo'):
            delattr(self, 'photo')
        
        if self.calibration_window:
            self.calibration_window.destroy()
            self.calibration_window = None