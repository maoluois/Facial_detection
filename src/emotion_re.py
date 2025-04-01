import cv2
import mediapipe as mp
import dlib
import imutils
import numpy as np
import torchlm
import os
import pickle
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
from datetime import datetime
import time
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

class EmotionRecognition:
    def __init__(self):
        # 初始化Dlib人脸检测器和特征点检测器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
        
        # MediaPipe面网格用于更精细的特征点提取
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 历史数据队列
        self.head_pose_history = deque(maxlen=10)  # 减少历史数据量以提高性能
        self.micro_expression_history = []
        
        # 3D人脸模型关键点 - 用于头部姿态估计
        self.model_points_68 = self._get_full_model_points()
        
        # 相机内参（估计值）
        self.camera_matrix = np.array(
            [[840, 0, 320],
             [0, 840, 240],
             [0, 0, 1]], dtype=np.float64
        )
        self.dist_coeffs = np.zeros((4, 1))
        
        # 状态变量
        self.last_time = time.time()
        self.last_landmarks = None
        self.emotion_confidence = {
            "Focused": 0,
            "Distracted": 0,
            "Confused": 0,
            "Fatigued": 0,
            "Excited": 0
        }
        
        self.ear_history = deque(maxlen=5)
        self.last_smooth_ear = None
        self.alpha = 0.25  # 平滑系数
        
        print("表情识别模块已初始化 v1.0")
    
    def _get_full_model_points(self):
        """获取68点人脸模型的3D关键点"""
        model_points = np.array([
            (0.0, 0.0, 0.0),                  # 鼻尖 - 30
            (0.0, -330.0, -65.0),             # 下巴 - 8
            (-225.0, 170.0, -135.0),          # 左眼左角 - 36
            (225.0, 170.0, -135.0),           # 右眼右角 - 45
            (-150.0, -150.0, -125.0),         # 左嘴角 - 48
            (150.0, -150.0, -125.0),          # 右嘴角 - 54
            (-170.0, 170.0, -135.0),          # 左眉毛外侧 - 17
            (170.0, 170.0, -135.0),           # 右眉毛外侧 - 26
            (-100.0, 200.0, -135.0),          # 左眉毛内侧 - 21
            (100.0, 200.0, -135.0),           # 右眉毛内侧 - 22
            (0.0, 100.0, -30.0),              # 鼻子底部 - 33
            (-60.0, -100.0, -80.0),           # 嘴左上角 - 60
            (60.0, -100.0, -80.0),            # 嘴右上角 - 64
            (0.0, -120.0, -90.0)              # 嘴下角 - 57
        ])
        return model_points
    
    def get_landmarks_dlib(self, image):
        """使用dlib获取68个面部关键点"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            return None
        
        # 获取第一个检测到的人脸的关键点
        shape = self.predictor(gray, faces[0])
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
        
        return landmarks, faces[0]
    
    def estimate_head_pose(self, landmarks, image):
        """使用68点特征点估计头部姿态"""
        if landmarks is None or len(landmarks) < 68:
            return None, None, None, None, None
        
        # 选择用于PnP解算的点
        image_points = np.array([
            landmarks[30],    # 鼻尖
            landmarks[8],     # 下巴
            landmarks[36],    # 左眼左角
            landmarks[45],    # 右眼右角
            landmarks[48],    # 左嘴角
            landmarks[54],    # 右嘴角
            landmarks[17],    # 左眉毛外侧
            landmarks[26],    # 右眉毛外侧
            landmarks[21],    # 左眉毛内侧
            landmarks[22],    # 右眉毛内侧
            landmarks[33],    # 鼻子底部
            landmarks[60],    # 嘴左上角
            landmarks[64],    # 嘴右上角
            landmarks[57],    # 嘴下角
        ], dtype=np.float64)
        
        # 求解PnP问题
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points_68, image_points, self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE)
        
        if not success:
            return None, None, None, None, None
        
        # 转换旋转向量为欧拉角
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat([rotation_mat, translation_vec])
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        
        pitch, yaw, roll = [angle[0] for angle in euler_angles]
        
        # 保存头部姿态历史
        self.head_pose_history.append((pitch, yaw, roll))
        
        return pitch, yaw, roll, rotation_vec, translation_vec
    
    def detect_action_units(self, landmarks):
        """检测面部动作单元(AUs)"""
        if landmarks is None or len(landmarks) < 68:
            return {}
        
        # 计算各部位的几何特征
        aus = {}
        
        # AU4: 皱眉 - 计算眉间距离变化
        inner_brow_distance = np.linalg.norm(landmarks[21] - landmarks[22])
        brow_height = (landmarks[21][1] + landmarks[22][1]) / 2 - landmarks[27][1]
        aus["AU4"] = brow_height < 5  # 皱眉时眉毛会下降
        
        # 检测左右皱眉的不对称性
        left_brow_height = landmarks[21][1] - landmarks[27][1]
        right_brow_height = landmarks[22][1] - landmarks[27][1]
        aus["AU4_asymmetry"] = abs(left_brow_height - right_brow_height) > 5
        
        # AU5: 上眼睑提升
        left_eye_height = np.linalg.norm(landmarks[37] - landmarks[41])
        right_eye_height = np.linalg.norm(landmarks[44] - landmarks[46])
        eye_height_avg = (left_eye_height + right_eye_height) / 2
        aus["AU5"] = eye_height_avg > 15  # 上眼睑提升时眼睛高度增加
        
        # AU6+AU7: 眯眼
        left_eye_ratio = left_eye_height / np.linalg.norm(landmarks[36] - landmarks[39])
        right_eye_ratio = right_eye_height / np.linalg.norm(landmarks[42] - landmarks[45])
        eye_ratio_avg = (left_eye_ratio + right_eye_ratio) / 2
        aus["AU6_7"] = eye_ratio_avg < 0.2  # 眯眼时高宽比降低
        
        # AU12: 嘴角上扬（微笑）
        mouth_corner_height = (landmarks[54][1] + landmarks[48][1]) / 2
        mouth_center_height = landmarks[57][1]
        aus["AU12"] = mouth_corner_height < mouth_center_height -18 # 微笑时嘴角上扬
        
        # AU25: 嘴唇分离（轻微张口）
        mouth_open = np.linalg.norm(landmarks[62] - landmarks[66])
        aus["AU25"] = mouth_open > 6 and mouth_open < 15  # 轻微张口
        
        # AU26+AU27: 下颌下降（打哈欠）
        jaw_drop = np.linalg.norm(landmarks[62] - landmarks[66])
        aus["AU26_27"] = jaw_drop > 25 and mouth_open > 25  # 打哈欠时下颌明显下降
        # 打印所有判据
        print(f"inner_brow_distance: {inner_brow_distance}, brow_height: {brow_height}, left_brow_height: {left_brow_height}, right_brow_height: {right_brow_height}, left_eye_height: {left_eye_height}, right_eye_height: {right_eye_height}, eye_height_avg: {eye_height_avg}, left_eye_ratio: {left_eye_ratio}, right_eye_ratio: {right_eye_ratio}, eye_ratio_avg: {eye_ratio_avg}, mouth_corner_height: {mouth_corner_height}, mouth_center_height: {mouth_center_height}, mouth_open: {mouth_open}, jaw_drop: {jaw_drop}")
    
        return aus

    # 添加计算眼睛长宽比（EAR）的函数
    def calculate_ear(self, eye_points):
        """计算眼睛的长宽比，用于眨眼检测"""
        # 计算垂直距离
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        # 计算水平距离
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        # 计算EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_gaze_direction(self, landmarks):
        """检测视线方向"""
        if landmarks is None or len(landmarks) < 68:
            return None
        
        # 眼球中心
        left_eye_center = np.mean([landmarks[36], landmarks[37], landmarks[38], 
                                   landmarks[39], landmarks[40], landmarks[41]], axis=0)
        right_eye_center = np.mean([landmarks[42], landmarks[43], landmarks[44], 
                                    landmarks[45], landmarks[46], landmarks[47]], axis=0)
        
        # 头部朝向（简化估计）
        face_center = landmarks[30]  # 鼻尖
        
        # 视线向量（从眼球中心到面部中心的反方向）
        gaze_vector = np.mean([left_eye_center, right_eye_center], axis=0) - face_center
        gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
        
        return gaze_vector
    
    def smooth_ear_value(self, landmarks):
        """计算平滑的眼睛纵横比值"""
        left_eye_ear = self.calculate_ear([landmarks[36], landmarks[37], landmarks[38], landmarks[39], landmarks[40], landmarks[41]])
        right_eye_ear = self.calculate_ear([landmarks[42], landmarks[43], landmarks[44], landmarks[45], landmarks[46], landmarks[47]])
        current_ear = (left_eye_ear + right_eye_ear) / 2
        
        # 异常值检测
        if self.ear_history and len(self.ear_history) >= 3:
            median_ear = sorted(list(self.ear_history))[len(self.ear_history)//2]
            if abs(current_ear - median_ear) > 0.15:
                current_ear = median_ear
        
        # 添加到历史队列
        self.ear_history.append(current_ear)
        
        # 应用EWMA
        if self.last_smooth_ear is None:
            avg_ear = current_ear
        else:
            avg_ear = self.alpha * current_ear + (1 - self.alpha) * self.last_smooth_ear
        
        # 更新上一帧平滑值
        self.last_smooth_ear = avg_ear
        
        return avg_ear

    
    def detect_expressions(self, image):
        """检测面部表情状态，返回适合Tkinter显示的结果"""
        # 创建图像副本以避免修改原始图像
        result_image = image.copy()
        
        # 获取面部关键点
        landmarks_and_face = self.get_landmarks_dlib(image)
        if landmarks_and_face is None:
            # 没有检测到人脸时，返回原始图像和提示信息
            cv2.putText(result_image, "No face detected", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return result_image
        
        landmarks, face_rect = landmarks_and_face
        
        # 估计头部姿态
        pitch, yaw, roll, rotation_vec, translation_vec = self.estimate_head_pose(landmarks, image)
        
        # 检测动作单元
        aus = self.detect_action_units(landmarks)
        
        # 检测视线方向
        gaze_vector = self.detect_gaze_direction(landmarks)
    
        # 计算眼睛开合度 (EAR - Eye Aspect Ratio)
        avg_ear = self.smooth_ear_value(landmarks)  
                
        # 计算动作变化（如果有之前的关键点）
        micro_expression = False
        movement_magnitude = 0
        if self.last_landmarks is not None:
            # 计算面部关键点变化
            movement = np.linalg.norm(landmarks - self.last_landmarks, axis=1)
            movement_magnitude = np.mean(movement)
            
            # 微表情判断更精细化
            mouth_movement = np.mean(movement[48:68])  # 嘴部区域移动
            eye_movement = np.mean(movement[36:48])   # 眼睛区域移动
            brow_movement = np.mean(movement[17:27])  # 眉毛区域移动
            
            # 微表情是局部的小幅度变化
            if (1 < mouth_movement < 3 or 
                0.5 < eye_movement < 2 or 
                0.5 < brow_movement < 2):
                micro_expression = True
        
        self.last_landmarks = landmarks.copy()  # 保存当前帧的关键点
        
        # 检测头部稳定性
        head_stable = True
        if len(self.head_pose_history) > 5:
            recent_yaws = [pose[1] for pose in list(self.head_pose_history)[-5:]]
            recent_pitches = [pose[0] for pose in list(self.head_pose_history)[-5:]]
            yaw_variation = np.std(recent_yaws)
            pitch_variation = np.std(recent_pitches)
            head_stable = yaw_variation < 5 and pitch_variation < 5
        
        # 重置信心值，使用指数衰减
        for emotion in self.emotion_confidence:
            self.emotion_confidence[emotion] *= 0.75 # 提高保留率，使情绪状态更稳定
        
        # ===== 表情状态判断条件 =====
        
        # 1. 专注状态（Focused）
        focused_conditions = [  
            (yaw is not None and abs(yaw) < 20),          #  头部朝向前方
            (avg_ear > 0.22 and avg_ear < 0.26),          #  眼睛睁开度适中
            head_stable,                                  #  头部稳定
            not micro_expression,                         #  没有微表情干扰
            aus.get("AU4", False),                        #  专注皱眉
        ]
        focused_score = sum([10, 12, 7, 5, 5][i] for i in range(len(focused_conditions)) if focused_conditions[i])
        self.emotion_confidence["Focused"] += min(35, focused_score)
        
        # 2. 分心状态（Distraction）
        head_turning = (yaw is not None and abs(yaw) > 25)  
        frequent_movements = movement_magnitude > 3         
        micro_expr_move = (micro_expression and movement_magnitude > 2) 

        distracted_score = 0
        if head_turning:
            distracted_score += 10
        if frequent_movements:
            distracted_score += 10
        if micro_expr_move:
            distracted_score += 15

        self.emotion_confidence["Distracted"] += min(30, distracted_score)
        
        # 3. 困惑状态（Confusion）
        confusion_conditions = [
            aus.get("AU4_asymmetry", False),   #  单侧皱眉            
            aus.get("AU6_7", False),           #  眯眼
            aus.get("AU25", False),            #  嘴唇微开
            (0.16 < avg_ear < 0.20 and not aus.get("AU12", False)) #  眼睛半睁
        ]
        confusion_score = sum([10, 5, 7, 15][i] for i in range(len(confusion_conditions)) if confusion_conditions[i])
        self.emotion_confidence["Confused"] += min(30, confusion_score)
    
        # 4. 疲劳状态（Fatigue）
        fatigue_conditions = [
            (pitch is not None and pitch < -10), #  头部下倾
            (avg_ear < 0.17 and not aus.get("AU12", False)),    #  眼睛半闭
            aus.get("AU6_7", False),             #  眯眼
            aus.get("AU26_27", False)            #  打哈欠
        ]
        fatigue_score = sum([10, 12, 5, 12][i] for i in range(len(fatigue_conditions)) if fatigue_conditions[i])
        self.emotion_confidence["Fatigued"] += min(35, fatigue_score)
        
        # 5. 兴奋状态（Excitement）
        excitement_conditions = [
            aus.get("AU12", False),     #  微笑
            aus.get("AU25", False),     #  嘴唇微开
            avg_ear > 0.27,             #  眼睛睁大
            movement_magnitude > 1      #  面部活跃
        ]
        excitement_score = sum([15, 7, 12, 5][i] for i in range(len(excitement_conditions)) if excitement_conditions[i])
        self.emotion_confidence["Excited"] += min(35, excitement_score)
            
        # 确定最高可能的情绪状态
        max_emotion = max(self.emotion_confidence, key=self.emotion_confidence.get)
        max_confidence = self.emotion_confidence[max_emotion]
        
        # 如果信心值太低，显示为未知
        if max_confidence < 60:
            max_emotion = "Unknown"
        
        # 绘制面部关键点
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(result_image, (x, y), 1, (0, 255, 0), -1)
        
        # 绘制面部边界框
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # 显示头部姿态
        if rotation_vec is not None and translation_vec is not None:
            # 计算头部前方的点以显示朝向
            nose_end_point2D = cv2.projectPoints(
                np.array([(0, 0, 200.0)]), rotation_vec, translation_vec, 
                self.camera_matrix, self.dist_coeffs)[0]
            
            p1 = (int(landmarks[30][0]), int(landmarks[30][1]))  # 鼻尖
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            
            cv2.line(result_image, p1, p2, (0, 0, 255), 2)
        
        # 显示表情状态
        cv2.putText(result_image, f"Emotion: {max_emotion}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # 显示信心度
        cv2.putText(result_image, f"Confidence: {max_confidence:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # 显示检测到的AU特征
        active_aus = [au for au, active in aus.items() if active]
        if active_aus:
            cv2.putText(result_image, f"AUs: {', '.join(active_aus)}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # 添加眼睛开合度显示
        cv2.putText(result_image, f"Eye Ratio: {avg_ear:.3f}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return result_image

