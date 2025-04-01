import cv2
import numpy as np
import time
from collections import deque
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure  # 添加这一行导入
from datetime import datetime
from face_recognition import FaceRecognition
from emotion_recognition import EmotionRecognition

class ComprehensionEstimator:
    def __init__(self, history_length=10):
        """
        初始化理解度估计器（使用卡尔曼滤波）
        :param history_length: 存储历史理解度分数的最大长度
        """
        self.history_length = history_length
        self.understanding_history = {}  # 存储每个学生的理解度历史

        # 卡尔曼滤波参数
        self.Q = 6e-8   # 过程噪声协方差 (越大代表理解度变化剧烈)
        self.R = 1e-4   # 测量噪声协方差 (越小代表分数测量更精确)
        self.P = 1.0    # 估计协方差
        self.x = 0   # 初始估计值 

    def smooth_understanding_score(self, student_id, current_score):
        """使用卡尔曼滤波平滑理解度分数"""
        if student_id not in self.understanding_history:
            self.understanding_history[student_id] = {
                "x": self.x,  # 初始状态
                "P": self.P,  # 估计协方差
            }
        
        # 读取当前学生的卡尔曼状态
        state = self.understanding_history[student_id]
        x, P = state["x"], state["P"]
        
        # === 预测步骤 ===
        x_predict = x  # 预测的理解度值（假设理解度不会剧烈变化）
        P_predict = P + self.Q  # 预测的不确定性增加
        
        # === 更新步骤 ===
        K = P_predict / (P_predict + self.R)  # 计算卡尔曼增益
        x_update = x_predict + K * (current_score - x_predict)  # 更新理解度估计
        P_update = (1 - K) * P_predict  # 更新估计协方差
        
        # 存储更新后的状态
        self.understanding_history[student_id]["x"] = x_update
        self.understanding_history[student_id]["P"] = P_update

        return x_update
    
    
class ClassroomMonitor:
    """学生课堂状态监测系统"""
    
    def __init__(self):
        # 初始化人脸识别和表情识别模块
        self.face_recognition = FaceRecognition()
        self.emotion_recognition = EmotionRecognition()
              
        # 初始化理解度估计器
        self.comprehension_estimator = ComprehensionEstimator()

        # 添加数据收集结构
        self.student_understanding_data = {}  # 格式: {student_id: [(timestamp, score), ...]}
        self.session_start_time = None
        self.frame_count = 0  # 帧计数器，用于控制数据采样频率
        self.sampling_rate = 1  # 每X帧采样一次数据
        
        # 理解度历史记录
        self.understanding_history = {}  # 按学生ID存储
        self.history_length = 10         # 历史记录长度
        
        # 平滑系数
        self.smoothing_alpha = 0.3
        
        # 界面设置
        self.display_font = cv2.FONT_HERSHEY_SIMPLEX
        self.primary_color = (50, 205, 50)     # 绿色
        self.alert_color = (0, 0, 255)         # 红色
        self.neutral_color = (255, 255, 255)   # 白色
        self.header_color = (255, 215, 0)      # 金色
        
        # 状态标签映射
        self.emotion_labels = {
            "Focused": "Focused",
            "Distracted": "Distracted",
            "Confused": "Confused",
            "Fatigued": "Fatigued", 
            "Excited": "Excited",
            "Unknown": "Unknown"
        }
        
        # 实时趋势图相关
        self.figure = None
        self.ax = None
        self.lines = {}  # 存储每个学生的曲线对象
        self.colors = plt.cm.tab10.colors  # 预定义10种颜色
        
        print("课堂状态监测模块已初始化 v1.0")
    
    def calculate_understanding_score(self, emotion_scores):
        """
        基于五种情绪状态计算学生的课程理解度评分
        """
        # 设定各情绪状态对理解度的权重系数
        weights = {
            "Focused": 0.50,     # 专注是理解的最大正向因素
            "Distracted": -0.40, # 分心严重影响理解
            "Confused": 0.00,   # 困惑表示理解障碍，但可能是思考过程
            "Fatigued": -0.20,   # 疲劳降低认知能力
            "Excited": 0.10      # 适度兴奋有助于理解和记忆
        }
        
        # 计算加权得分
        weighted_score = 0
        for emotion, score in emotion_scores.items():
            if emotion in weights:
                weighted_score += weights[emotion] * score
        
        final_score = weighted_score
        
        return final_score
    
    def smooth_understanding_score(self, student_id, current_score):
        """平滑处理理解度分数，避免剧烈波动"""
        return self.comprehension_estimator.smooth_understanding_score(student_id, current_score)
    
    # def smooth_understanding_score(self, student_id, current_score):
    #     """平滑处理理解度分数，避免剧烈波动"""
    #     if student_id not in self.understanding_history:
    #         self.understanding_history[student_id] = deque(maxlen=self.history_length)
            
    #     history = self.understanding_history[student_id]
    #     history.append(current_score)
        
    #     # 如果历史记录不足，直接返回当前分数
    #     if len(history) < 3:
    #         return current_score
            
    #     # 应用指数加权移动平均
    #     alpha = self.smoothing_alpha
    #     smoothed_score = current_score * alpha + (1 - alpha) * sum(list(history)[:-1]) / (len(history) - 1)
        
    #     return smoothed_score
    
    def process_frame(self, frame):
        """处理视频帧，返回增强显示的帧"""
        # 如果是第一帧，记录开始时间
        if self.session_start_time is None:
            self.session_start_time = time.time()
        
        self.frame_count += 1
        result_image = frame.copy()
        
        # 获取面部关键点和边界框
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_recognition.detector(gray, 0)
        
        if len(faces) == 0:
            cv2.putText(result_image, "No student detect", (20, 50), 
                      self.display_font, 1.0, self.alert_color, 2)
            return result_image
        
        # 添加状态监测标题
        cv2.putText(result_image, "student_monitor_system", (20, 30),
                    self.display_font, 1.0, self.header_color, 2)
        
        # 在每个检测到的人脸上执行分析
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        
            # 1. 使用优化后的人脸识别方法
            student_id, confidence, landmarks = self.face_recognition.identify_face(frame, face, gray)

            # # 计算眼睛的开合度
            # avg_ear = self.face_recognition.smoother_ear(landmarks)
            
            # 2. 从整帧中裁剪出人脸区域
            face_image = frame[max(0, y1-30):min(frame.shape[0], y2+30), 
                            max(0, x1-30):min(frame.shape[1], x2+30)]
            
            # 确保裁剪区域有效
            if face_image.size == 0:
                face_image = frame[y1:y2, x1:x2]  # 使用更小的区域
            
            if face_image.size > 0:
                # 3. 对每个人脸单独进行情绪检测
                self.emotion_recognition.detect_expressions(face_image, student_id)
                
                # 获取该人脸的情绪分数
                emotion_scores = self.emotion_recognition.emotion_confidence
            else:
                # 如果裁剪失败，使用默认的情绪分数
                emotion_scores = {"Focused": 0, "Distracted": 0, "Confused": 0, 
                                "Fatigued": 0, "Excited": 0}
            
            # 识别主要情绪 - 使用自定义阈值
            max_emotion = max(emotion_scores, key=emotion_scores.get)
            max_confidence = emotion_scores[max_emotion]
            
            if max_confidence < 1:
                max_emotion = "Unknown"
            
            emotion_label = self.emotion_labels.get(max_emotion, "Unknown")
            
            # 3. 计算理解度
            understanding_score = self.calculate_understanding_score(emotion_scores)
            smoothed_score = self.smooth_understanding_score(student_id, understanding_score)
        
            # 收集数据 - 每X帧采样一次，避免数据过多
            if self.frame_count % self.sampling_rate == 0 and student_id != "Unknown":
                current_time = time.time() - self.session_start_time  # 相对时间(秒)
                
                if student_id not in self.student_understanding_data:
                    self.student_understanding_data[student_id] = []
                
                self.student_understanding_data[student_id].append((current_time, smoothed_score))
            
            # 绘制结果
            # 1. 面部边界框
            box_color = self.primary_color if max_emotion == "Focused" else (
                         self.alert_color if max_emotion in ["Distracted", "Fatigued"] else 
                         self.neutral_color)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), box_color, 2)
            
            # 计算信息框位置
            info_x = min(x1, frame.shape[1] - 200)
            info_y = min(y2 + 10, frame.shape[0] - 90)
            
            # 绘制信息框
            cv2.rectangle(result_image, (info_x, info_y), (info_x + 180, info_y + 85), (45, 45, 45), -1)
            cv2.rectangle(result_image, (info_x, info_y), (info_x + 180, info_y + 85), box_color, 2)
            
            # 只显示三项关键信息
            cv2.putText(result_image, f"ID: {student_id}", (info_x + 10, info_y + 20), 
                      self.display_font, 0.5, (255, 255, 255), 1)
            cv2.putText(result_image, f"Emotion: {emotion_label}", (info_x + 10, info_y + 45), 
                      self.display_font, 0.5, self.neutral_color, 1)
            cv2.putText(result_image, f"Score: {smoothed_score:.1f}", (info_x + 10, info_y + 70), 
                      self.display_font, 0.5, self.neutral_color, 1)
                    # 添加眼睛开合度显示
            # cv2.putText(result_image, f"Eye Ratio: {avg_ear:.3f}", (10, 120), 
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return result_image
    
    def generate_understanding_reports(self, output_directory="reports"):
        """仅生成每个学生的理解度趋势图"""
        # 创建输出目录
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        # 生成报告时间戳
        report_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_count = 0
        
        # 为每个学生生成趋势图
        for student_id, data_points in self.student_understanding_data.items():
            # 跳过数据点太少的学生
            if len(data_points) < 5:
                continue
                
            # 解析数据
            timestamps = [point[0] for point in data_points]
            scores = [point[1] for point in data_points]
            
            # 创建图表
            plt.figure(figsize=(10, 6))
            plt.plot(timestamps, scores, 'b-', linewidth=2)
            plt.fill_between(timestamps, 0, scores, alpha=0.2)
            
            # 添加标题和标签
            plt.title(f"Student Understanding Trend - {student_id}")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Understanding Score")
            plt.ylim(0, max(100, max(scores) + 10))
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 添加统计信息
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            stats_text = f"Avg: {avg_score:.1f}\nMax: {max_score:.1f}\nMin: {min_score:.1f}"
            plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                        verticalalignment='top')
            
            # 保存图表
            filename = f"{output_directory}/{report_datetime}_{student_id}.png"
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            plt.close()
            
            report_count += 1
        
        return report_count
    
    def setup_realtime_chart(self):
        """初始化实时趋势图"""
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        
        # 设置图表属性
        self.ax.set_title("Real-time Understanding Trends")
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Understanding Score")
        self.ax.set_ylim(-20, 20)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        return self.figure
        
    def update_realtime_chart(self):
        """更新实时趋势图"""
        if self.figure is None or self.ax is None:
            return
            
        # 清除当前的图例
        if self.ax.get_legend() is not None:
            self.ax.get_legend().remove()
            
        # 根据最新数据更新每个学生的曲线
        for i, (student_id, data_points) in enumerate(self.student_understanding_data.items()):
            if len(data_points) < 2:  # 至少需要两个点才能画线
                continue
                
            timestamps = [point[0] for point in data_points]
            scores = [point[1] for point in data_points]
            
            # 如果这个学生还没有曲线，创建新曲线
            if student_id not in self.lines:
                color_idx = len(self.lines) % len(self.colors)
                line, = self.ax.plot(timestamps, scores, 
                                    color=self.colors[color_idx], 
                                    label=f"Student {student_id}",
                                    linewidth=2)
                self.lines[student_id] = line
            else:
                # 更新已有曲线
                self.lines[student_id].set_data(timestamps, scores)
        
        # 动态调整X轴范围
        if len(self.student_understanding_data) > 0:
            all_times = [point[0] for data in self.student_understanding_data.values() for point in data]
            if all_times:
                max_time = max(all_times)
                self.ax.set_xlim(0, max(30, max_time * 1.1))  # 留出一些空间
                
        # 添加图例
        if self.lines:
            self.ax.legend(loc='upper right')
            
        # 刷新图表
        self.figure.canvas.draw_idle()
    