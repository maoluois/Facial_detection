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
import dlib

if dlib.DLIB_USE_CUDA:
    print("dlib is using GPU")
else:
    print("dlib is using CPU")
# print("DLIB_USE_CUDA:", dlib.DLIB_USE_CUDA)
# print("CUDA version:", dlib.cuda.get_version())
# print("Number of CUDA devices:", dlib.cuda.get_num_devices())
# print("Current CUDA device:", dlib.cuda.get_device())

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

norm_eye = {"lmz" : 0.33, "phr" : 0.24}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def normalize(value, min_val, max_val):
    """将数值归一化到 [0, 1]"""
    return (value - min_val) / (max_val - min_val)

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
            [[6894, 0, 301],
             [0, 6901, 230],
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
        self.smoothing_emotion_alpha = 1 # 表情平滑系数      
        self.alpha = 0.75  # 平滑系数
        self.normalization_ranges = {                # 归一化标准值字典
            "yaw_forward": (-20, 20),  # 头部前倾
            "yaw_turn": (0, 25),  # 头部转向
            "movement": (0, 5),  # 运动幅度
            "eye_height": (6, 12),  # 眼睛高度
            "eye_ratio": (0.2, 0.5),  # 眼睛开合比例
            "brow_height": (-10, 10),  # 眉毛高度
            "brow_asymmetry": (0, 10),  # 眉毛不对称度
            "mouth_open": (5, 25),  # 嘴巴张开
            "jaw_drop": (15, 40),  # 下巴下垂
            "ear": (0.15, 0.3),  # 眼睛长宽比 EAR
        }

        
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
    
    

    # 添加计算眼睛长宽比（EAR）的函数
    def calculate_ear(self, eye_points, ID):
        """计算眼睛的长宽比，用于眨眼检测"""
        # 计算垂直距离
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        # 计算水平距离
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        # 计算EAR
        ear = (A + B) / (2.0 * C)

        if ID in norm_eye:
            ear = ear * (0.24 / norm_eye[ID])
                
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
    
    def smooth_ear_value(self, landmarks, ID):
        """计算平滑的眼睛纵横比值"""
        left_eye_ear = self.calculate_ear([landmarks[36], landmarks[37], landmarks[38], landmarks[39], landmarks[40], landmarks[41]], ID)
        right_eye_ear = self.calculate_ear([landmarks[42], landmarks[43], landmarks[44], landmarks[45], landmarks[46], landmarks[47]], ID)
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

    
    def detect_expressions(self, image, ID):
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
        
        # # 检测动作单元
        # aus = self.detect_action_units(landmarks)
        
        # 检测视线方向
        gaze_vector = self.detect_gaze_direction(landmarks)
    
        # 计算眼睛开合度 (EAR - Eye Aspect Ratio)
        avg_ear = self.smooth_ear_value(landmarks, ID)  
                
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
        

    # def detect_action_units(self, landmarks):
    #     """检测面部动作单元(AUs)"""
        # if landmarks is None or len(landmarks) < 68:
        #     return {}

        # 计算各部位的几何特征
        aus = {}

        # 头部方向
        aus['head_forward'] = max(0, 1 - abs(normalize(yaw, *self.normalization_ranges["yaw_forward"]))) if yaw is not None else 0
        aus['head_turn'] = normalize(abs(yaw), *self.normalization_ranges["yaw_turn"]) if yaw is not None else 0
        print("yaw")
        # 面部运动
        aus['frequent_movement'] = normalize(movement_magnitude, *self.normalization_ranges["movement"])
        aus['face_active'] = normalize(movement_magnitude * 1.5, *self.normalization_ranges["movement"])  # 放大运动活跃度
        aus['no_micro_expression'] = 1 - int(micro_expression)

        # 眉毛特征
        brow_height = (landmarks[21][1] + landmarks[22][1]) / 2 - landmarks[27][1]
        left_brow_height = landmarks[21][1] - landmarks[27][1]
        right_brow_height = landmarks[22][1] - landmarks[27][1]

        aus["BrowFurrow"] = sigmoid(normalize(brow_height, *self.normalization_ranges["brow_height"]))
        aus["BrowFurrowAsymmetry"] = sigmoid(normalize(abs(left_brow_height - right_brow_height), *self.normalization_ranges["brow_asymmetry"]))

        # 眼睛特征
        left_eye_height = np.linalg.norm(landmarks[37] - landmarks[41])
        right_eye_height = np.linalg.norm(landmarks[44] - landmarks[46])
        eye_height_avg = (left_eye_height + right_eye_height) / 2
        left_eye_ratio = left_eye_height / np.linalg.norm(landmarks[36] - landmarks[39])
        right_eye_ratio = right_eye_height / np.linalg.norm(landmarks[42] - landmarks[45])
        eye_ratio_avg = (left_eye_ratio + right_eye_ratio) / 2

        aus["UpperLidRaiser"] = sigmoid(normalize(eye_height_avg, *self.normalization_ranges["eye_height"]))
        aus["EyeSquint"] = 1 - sigmoid((normalize(eye_ratio_avg, *self.normalization_ranges["eye_ratio"]) - 0.35) * 10)

        # 嘴部特征
        mouth_corner_height = (landmarks[54][1] + landmarks[48][1]) / 2
        mouth_center_height = landmarks[57][1]
        mouth_open = np.linalg.norm(landmarks[62] - landmarks[66])
        jaw_drop = np.linalg.norm(landmarks[62] - landmarks[66])

        aus["Smile"] = sigmoid(normalize((mouth_center_height - mouth_corner_height), *self.normalization_ranges["brow_height"]))
        aus["LipsPart"] = sigmoid(normalize(mouth_open, *self.normalization_ranges["mouth_open"])) - sigmoid(normalize(mouth_open - 10, *self.normalization_ranges["mouth_open"]))
        aus["JawDrop"] = sigmoid(normalize(jaw_drop, *self.normalization_ranges["jaw_drop"]))



        # print(f"brow_height: {brow_height}, left_brow_height: {left_brow_height}, right_brow_height: {right_brow_height}, left_eye_height: {left_eye_height}, right_eye_height: {right_eye_height}, eye_height_avg: {eye_height_avg}, left_eye_ratio: {left_eye_ratio}, right_eye_ratio: {right_eye_ratio}, eye_ratio_avg: {eye_ratio_avg}, mouth_corner_height: {mouth_corner_height}, mouth_center_height: {mouth_center_height}, mouth_open: {mouth_open}, jaw_drop: {jaw_drop}")
        # print(aus)    
        # return aus    
        
        # ===== 表情状态判断条件 =====
        
        focused_conditions = [
        aus["head_forward"],
        normalize(avg_ear, 0.22, 0.26),
        int(head_stable),
        aus["no_micro_expression"],
        aus["BrowFurrow"],
        ]
        focused_score = sum([10, 5, 10, 5, 5][i] * focused_conditions[i] for i in range(len(focused_conditions)))

        # 2. 分心状态（Distraction）
        distraction_conditions = [
            aus["head_turn"],
            aus["frequent_movement"],
            micro_expression * aus["face_active"],
        ]
        distracted_score = sum([15, 10, 10][i] * distraction_conditions[i] for i in range(len(distraction_conditions)))

        # 3. 困惑状态（Confusion）
        confusion_conditions = [
            aus["BrowFurrowAsymmetry"],
            aus["EyeSquint"],
            aus["LipsPart"],
            normalize(avg_ear, 0.19, 0.22) * (1 - aus["Smile"]),
        ]
        confusion_score = sum([15, 5, 5, 10][i] * confusion_conditions[i] for i in range(len(confusion_conditions)))

        # 4. 疲劳状态（Fatigue）
        fatigue_conditions = [
            min(1, max(0, (pitch + 10) / 20)),  # 头部下倾归一化
            normalize(avg_ear, 0.0, 0.19) * (1 - aus["Smile"]),
            aus["EyeSquint"],
            aus["JawDrop"],
        ]
        fatigue_score = sum([5, 10, 5, 15][i] * fatigue_conditions[i] for i in range(len(fatigue_conditions)))

        # 5. 兴奋状态（Excitement）
        excitement_conditions = [
            aus["Smile"],
            aus["LipsPart"],
            normalize(avg_ear, 0.26, 0.35),
            aus["face_active"],
        ]
        excitement_score = sum([15, 5, 10, 5][i] * excitement_conditions[i] for i in range(len(excitement_conditions)))
        
        # 重置信心值，使用指数衰减
        for emotion in self.emotion_confidence:
            self.emotion_confidence[emotion] *= 0.75 # 提高保留率，使情绪状态更稳定

        # 更新情绪置信度，使用指数平滑
        for emotion in self.emotion_confidence:
            previous_confidence = self.emotion_confidence[emotion]
            new_score = 0  # 默认新得分

            if emotion == "Focused":
                new_score = min(50, focused_score)
            elif emotion == "Distracted":
                new_score = min(50, distracted_score)
            elif emotion == "Confused":
                new_score = min(50, confusion_score)
            elif emotion == "Fatigued":
                new_score = min(50, fatigue_score)
            elif emotion == "Excited":
                new_score = min(50, excitement_score)

            # 应用指数平滑
            self.emotion_confidence[emotion] = self.smoothing_emotion_alpha * new_score + (1 - self.smoothing_emotion_alpha) * previous_confidence
    
        # 确定最高可能的情绪状态
        max_emotion = max(self.emotion_confidence, key=self.emotion_confidence.get)
        max_confidence = self.emotion_confidence[max_emotion]
        # 如果信心值太低，显示为未知
        if max_confidence < 1:
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
        
        # print(f"Eye Ratio: {avg_ear:.3f}, gaze_vector: [{gaze_vector}], " + ", ".join([f"{emotion}: {confidence:.2f}" for emotion, confidence in self.emotion_confidence.items()]))
        print(aus)
        return result_image

