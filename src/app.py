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
        # 捕捉视频帧并计算AUs的最大值、最小值和标准值
        pass

    def save_calibration_data(self):
        # 保存校准数据到文件或变量中
        # 直接取中间值是否正确？？ ！！！！！！
        for au, data in self.calibration_data.items():
            data["standard"] = (data["max"] + data["min"]) / 2
    
        # 假设有一个函数 `save_to_file` 保存数据到文件
        self.save_to_file(self.calibration_data, "calibration_data.json")

    def save_to_file(self, data, filename):
        # 保存数据到文件
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)    

    def detect_aus(self, frame):
        pass
               
            
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