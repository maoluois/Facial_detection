import cv2
import dlib
import imutils
import numpy as np
import gradio as gr
import time

class FaceOrientation(object):
    def __init__(self):
        self.detect = dlib.get_frontal_face_detector()
        self.predict = dlib.shape_predictor(r"/home/lmz/cv_advanced/FaceProcessingWebcam/FaceAnalysisWebApp/model/dlib-models/shape_predictor_68_face_landmarks.dat")

    def process_frame(self, frame):
        draw_rect1 = True
        draw_rect2 = True
        draw_lines = True

        # 加载JPEG图像
        overlay_img = cv2.imread(r'/home/lmz/cv_advanced/OIP-C.jpg')
        h, w, _ = overlay_img.shape

        frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = self.detect(gray, 0)

        for subject in subjects:
            landmarks = self.predict(gray, subject)
            size = frame.shape

            # 2D image points. If you change the image, you need to change vector
            image_points = np.array([
                (landmarks.part(33).x, landmarks.part(33).y),  # Nose tip
                (landmarks.part(8).x, landmarks.part(8).y),  # Chin
                (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
                (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corne
                (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
                (landmarks.part(54).x, landmarks.part(54).y)  # Right mouth corner
            ], dtype="double")

            # 3D model points.
            model_points = np.array([
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (-225.0, 170.0, -135.0),  # Left eye left corner
                (225.0, 170.0, -135.0),  # Right eye right corne
                (-150.0, -150.0, -125.0),  # Left Mouth corner
                (150.0, -150.0, -125.0)  # Right mouth corner

            ])
            # Camera internals
            focal_length = size[1]
            center = (size[1] / 2, size[0] / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )

            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                          dist_coeffs)

            (b1, jacobian) = cv2.projectPoints(np.array([(350.0, 270.0, 0.0)]), rotation_vector, translation_vector,
                                               camera_matrix, dist_coeffs)
            (b2, jacobian) = cv2.projectPoints(np.array([(-350.0, -270.0, 0.0)]), rotation_vector,
                                               translation_vector, camera_matrix, dist_coeffs)
            (b3, jacobian) = cv2.projectPoints(np.array([(-350.0, 270, 0.0)]), rotation_vector, translation_vector,
                                               camera_matrix, dist_coeffs)
            (b4, jacobian) = cv2.projectPoints(np.array([(350.0, -270.0, 0.0)]), rotation_vector,
                                               translation_vector, camera_matrix, dist_coeffs)

            (b11, jacobian) = cv2.projectPoints(np.array([(450.0, 350.0, 400.0)]), rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)
            (b12, jacobian) = cv2.projectPoints(np.array([(-450.0, -350.0, 400.0)]), rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)
            (b13, jacobian) = cv2.projectPoints(np.array([(-450.0, 350, 400.0)]), rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)
            (b14, jacobian) = cv2.projectPoints(np.array([(450.0, -350.0, 400.0)]), rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)

            b1 = (int(b1[0][0][0]), int(b1[0][0][1]))
            b2 = (int(b2[0][0][0]), int(b2[0][0][1]))
            b3 = (int(b3[0][0][0]), int(b3[0][0][1]))
            b4 = (int(b4[0][0][0]), int(b4[0][0][1]))

            b11 = (int(b11[0][0][0]), int(b11[0][0][1]))
            b12 = (int(b12[0][0][0]), int(b12[0][0][1]))
            b13 = (int(b13[0][0][0]), int(b13[0][0][1]))
            b14 = (int(b14[0][0][0]), int(b14[0][0][1]))

            if draw_rect1 == True:
                cv2.line(frame, b1, b3, (255, 255, 0), 10)
                cv2.line(frame, b3, b2, (255, 255, 0), 10)
                cv2.line(frame, b2, b4, (255, 255, 0), 10)
                cv2.line(frame, b4, b1, (255, 255, 0), 10)

            if draw_rect2 == True:
                cv2.line(frame, b11, b13, (255, 255, 0), 10)
                cv2.line(frame, b13, b12, (255, 255, 0), 10)
                cv2.line(frame, b12, b14, (255, 255, 0), 10)
                cv2.line(frame, b14, b11, (255, 255, 0), 10)

            if draw_lines == True:
                cv2.line(frame, b11, b1, (0, 255, 0), 10)
                cv2.line(frame, b13, b3, (0, 255, 0), 10)
                cv2.line(frame, b12, b2, (0, 255, 0), 10)
                cv2.line(frame, b14, b4, (0, 255, 0), 10)

            # 定义图像的3D顶点（例如贴在人头的右边）
            overlay_3d_points = np.array([
                (-100.0, 550.0, 0.0),  # 顶点1
                (-100.0, 350.0, 0.0),  # 顶点2
                (100.0, 350.0, 0.0),  # 顶点3
                (100.0, 550.0, 0.0)   # 顶点4
            ], dtype="double")

            # 将3D顶点投影到2D图像平面
            (overlay_2d_points, jacobian) = cv2.projectPoints(overlay_3d_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            # 转换投影点坐标为整数
            overlay_2d_points = [(int(point[0][0]), int(point[0][1])) for point in overlay_2d_points]

            # 定义JPEG图像的四个顶点
            src_points = np.array([
                [0, 0],
                [0, h - 1],
                [w - 1, h - 1],
                [w - 1, 0]
            ], dtype="float32")

            # 定义目标图像的四个顶点
            dst_points = np.array(overlay_2d_points, dtype="float32")

            # 计算仿射变换矩阵
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)

            # 将JPEG图像贴到人头上
            warped_img = cv2.warpPerspective(overlay_img, matrix, (frame.shape[1], frame.shape[0]))

            # 创建一个掩码
            mask = np.zeros_like(frame, dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(dst_points), (255, 255, 255))

            # 将掩码应用到原始图像
            masked_frame = cv2.bitwise_and(frame, cv2.bitwise_not(mask))

            # 将变换后的图像叠加到原始图像上
            frame = cv2.add(masked_frame, warped_img)

        return frame

def process_video_frame(value):
    cap = cv2.VideoCapture(0)
    while 1:
        time.sleep(value) # 防止卡死
        ret, frame = cap.read()
        if not ret:
            break
        
        face_obj = FaceOrientation()
        face_obj.create_orientation()
        yield face_obj.process_frame(frame)

if __name__ == "__main__":
    iface = gr.Interface(
        fn=process_video_frame, 
        inputs=gr.Slider(label="调整输出图像帧的time.sleep的时长",minimum=0.01,maximum=0.1,step=0.005),
        outputs=[gr.Image(label="AR_vedio")],
        live=True
    )
    
    iface.queue
    iface.launch()