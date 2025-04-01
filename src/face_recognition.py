import cv2
import dlib
import numpy as np
import os
import pickle
from datetime import datetime
# import time
# from collections import deque
# import mediapipe as mp
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_face_mesh = mp.solutions.face_mesh
# mp_face_detection = mp.solutions.face_detection

class FaceRecognition:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        
        try:
            self.shape_predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
            self.face_rec_model = dlib.face_recognition_model_v1("model/dlib_face_recognition_resnet_model_v1.dat")
        except RuntimeError as e:
            print(f"错误：无法加载面部识别模型 - {e}")
            print("请确保在model目录中有以下文件:")
            print("- shape_predictor_68_face_landmarks.dat")
            print("- dlib_face_recognition_resnet_model_v1.dat")
        
        self.face_database = {}
        self.face_db_path = "data/face_database.pkl"
        
        self.load_faces()
        self.threshold = 0.5
        
        print("人脸识别模块已初始化")
    
    def load_faces(self):
        if os.path.exists(self.face_db_path):
            with open(self.face_db_path, 'rb') as f:
                self.face_database = pickle.load(f)
                print(f"已加载 {len(self.face_database)} 条人脸记录")
        else:
            self.face_database = {}
            print("未找到人脸数据库，创建新数据库")
    
    def save_faces(self):
        os.makedirs(os.path.dirname(self.face_db_path), exist_ok=True)
        with open(self.face_db_path, 'wb') as f:
            pickle.dump(self.face_database, f)
            print(f"已保存 {len(self.face_database)} 条人脸记录")
    
    def extract_face_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        
        if len(faces) == 0:
            return None, None, "未检测到人脸"
        
        face = faces[0]
        face_rect = (face.left(), face.top(), face.right(), face.bottom())
        landmarks = self.shape_predictor(gray, face)
        face_descriptor = self.face_rec_model.compute_face_descriptor(image, landmarks)
        face_descriptor = np.array(face_descriptor)
        
        return face_descriptor, face_rect, "人脸特征提取成功"
    
    def register_face(self, image, name):
        if not name:
            return image, "请输入人脸标识符"
        
        face_descriptor, face_rect, message = self.extract_face_features(image)
        
        if face_descriptor is None:
            return image, message
        
        self.face_database[name] = {
            'descriptor': face_descriptor, 
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.save_faces()
        
        result_image = image.copy()
        x1, y1, x2, y2 = face_rect
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result_image, f"已注册: {name}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result_image, f"人脸 '{name}' 注册成功"
    
    def recognize_face(self, image):
        face_descriptor, face_rect, message = self.extract_face_features(image)
        
        if face_descriptor is None:
            return image, message
        
        if len(self.face_database) == 0:
            return image, "人脸数据库为空，请先注册人脸"
        
        best_match = None
        best_distance = float('inf')
        
        for name, face_data in self.face_database.items():
            stored_descriptor = face_data['descriptor']
            distance = np.linalg.norm(face_descriptor - stored_descriptor)
            
            if distance < best_distance:
                best_distance = distance
                best_match = name
        
        result_image = image.copy()
        x1, y1, x2, y2 = face_rect
        
        if best_distance < self.threshold:
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_image, f"ID: {best_match}", 
                       (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result_image, f"置信度: {1-best_distance:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return result_image, f"ID:{best_match}, 置信度:{1 - best_distance:.2f}"
        else:
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(result_image, "未知", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return result_image, "未找到匹配的人脸"
    
    def list_registered_faces(self):
        if len(self.face_database) == 0:
            return "人脸数据库为空"
        
        result = "已注册的人脸:\n"
        for name, face_data in self.face_database.items():
            result += f"- {name} (注册时间: {face_data['timestamp']})\n"
        
        return result
    
    def delete_face(self, name):
        if name in self.face_database:
            del self.face_database[name]
            self.save_faces()
            return f"已删除 '{name}' 的人脸记录"
        else:
            return f"未找到 '{name}' 的人脸记录"
        
    def identify_face(self, frame, face, gray=None):
        """
        识别单个人脸，返回ID和置信度
        
        Parameters:
        - frame: 原始图像帧
        - face: dlib人脸检测结果
        - gray: 可选的灰度图像，如已计算可传入避免重复计算
        
        Returns:
        - (face_id, confidence): 识别到的人脸ID和置信度，未识别则ID为"Unknown"
        """
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        # 获取人脸特征
        landmarks = self.shape_predictor(gray, face)
        face_descriptor = self.face_rec_model.compute_face_descriptor(frame, landmarks)
        face_descriptor = np.array(face_descriptor)
        
        # 默认值
        face_id = "Unknown"
        confidence = 0
        
        # 数据库为空时直接返回默认值
        if len(self.face_database) == 0:
            return face_id, confidence, landmarks
        
        # 找最佳匹配
        best_match = None
        best_distance = float('inf')
        
        for name, face_data in self.face_database.items():
            stored_descriptor = face_data['descriptor']
            distance = np.linalg.norm(face_descriptor - stored_descriptor)
            
            if distance < best_distance:
                best_distance = distance
                best_match = name
        
        # 确认匹配结果
        if best_distance < self.threshold:
            face_id = best_match
            confidence = 1 - best_distance
            
        return face_id, confidence, landmarks
    
    def recognize_face_stream(self, frame):
        """为实时视频流添加人脸识别处理方法"""
        result_image = frame.copy()
        
        # 获取人脸特征
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)  # 使用0加速检测
        
        if len(faces) == 0:
            # 无人脸时返回原始帧
            return result_image
        
        # 处理每个检测到的人脸
        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            face_rect = (x1, y1, x2, y2)
            
            # 获取特征
            landmarks = self.shape_predictor(gray, face)
            face_descriptor = self.face_rec_model.compute_face_descriptor(frame, landmarks)
            face_descriptor = np.array(face_descriptor)
            
            if len(self.face_database) == 0:
                # 数据库为空时
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(result_image, "未知", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                continue
            
            # 找最佳匹配
            best_match = None
            best_distance = float('inf')
            
            for name, face_data in self.face_database.items():
                stored_descriptor = face_data['descriptor']
                distance = np.linalg.norm(face_descriptor - stored_descriptor)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = name
            
            # 显示结果
            if best_distance < self.threshold:
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(result_image, f"ID: {best_match}", 
                           (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result_image, f"Confidence Score: {1-best_distance:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(result_image, "Unknown ID", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result_image