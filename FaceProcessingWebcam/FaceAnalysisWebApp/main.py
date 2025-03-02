import cv2
import gradio as gr
import mediapipe as mp
import dlib
import imutils
import numpy as np
import torchlm
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet

# dlib探测不到人脸？？？ 人脸光亮，大小，模糊处理有影响？？
# cv设定摄像头参数？？？
# python3.10 需要chmod 777 gradio提示的插件
# scipy 使用1.13.0 高版本与torchlm有兼容性问题
# import torchvision.models as models
# from torchvision.models import ResNet18_Weights
# torchlm使用的是torchvision.models.resnet18(pretrained=True)模型，需要下载预训练权重
# dlib的shape_predictor需要下载预训练权重

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

def apply_media_pipe_face_detection(image):
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return image
        annotated_image = image.copy()
        for detection in results.detections:
            mp_drawing.draw_detection(annotated_image, detection)
        return annotated_image


def apply_media_pipe_facemesh(image):

    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return image
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
              image=annotated_image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_TESSELATION,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
              image=annotated_image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_CONTOURS,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
              image=annotated_image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_IRISES,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_iris_connections_style())
            return annotated_image


class FaceOrientation(object):
    def __init__(self):
        self.detect = dlib.get_frontal_face_detector()
        self.predict = dlib.shape_predictor("/home/luomao/Documents/Code/opencv/homework-00-maoluois-master/FaceProcessingWebcam/FaceAnalysisWebApp/model/shape_predictor_68_face_landmarks.dat")

    def create_orientation(self, frame):
        print("dlib start process image")
        draw_rect1 = True
        draw_rect2 = True
        draw_lines = True

        # 加载JPEG图像
        overlay_img = cv2.imread('/home/luomao/Documents/Code/opencv/homework-00-maoluois-master/OIP-C.jpg')
        h, w, _ = overlay_img.shape

        frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = self.detect(gray, 1)
        if len(subjects) == 0:
            print("No faces detected")
            return frame

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

            # ？？？
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
            # ？？？
            print("dlib finish process image")
        return frame

face_orientation_obj = FaceOrientation()

def apply_media_pipe_face_68(image):  
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
    return image

def apply_torchlm_face_68(image):
    torchlm.runtime.bind(faceboxesv2(device="cpu"))  # set device="cuda" if you want to run with CUDA
    torchlm.runtime.bind(
        pipnet(backbone="resnet18", pretrained=True, num_nb=10, num_lms=68, net_stride=32, input_size=256,
               meanface_type="300w", map_location="cpu", checkpoint="/home/luomao/Documents/Code/opencv/homework-00-maoluois-master/FaceProcessingWebcam/FaceAnalysisWebApp/model/pipnet_resnet18_10x68x32x256_300w.pth")  # set map_location="cuda" if you want to run with CUDA
    )
    landmarks, bboxes = torchlm.runtime.forward(image)
    image = torchlm.utils.draw_bboxes(image, bboxes=bboxes)
    image = torchlm.utils.draw_landmarks(image, landmarks=landmarks)
    return image

def apply_line_of_corresponding_points(image):
    torchlm.runtime.bind(faceboxesv2(device="cpu"))  # set device="cuda" if you want to run with CUDA
    torchlm.runtime.bind(
        pipnet(backbone="resnet18", pretrained=True, num_nb=10, num_lms=68, net_stride=32, input_size=256,
               meanface_type="300w", map_location="cpu", checkpoint="/home/luomao/Documents/Code/opencv/homework-00-maoluois-master/FaceProcessingWebcam/FaceAnalysisWebApp/model/pipnet_resnet18_10x68x32x256_300w.pth")  # set map_location="cuda" if you want to run with CUDA
    )
    
    # 处理 MediaPipe 模型
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,  # 只连接框住人脸的特征部分
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),  # 绿色点和线表示 MediaPipe 模型
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )
    
       # 处理 Torchlm 模型
    landmarks_torchlm, bboxes = torchlm.runtime.forward(image)
    if len(landmarks_torchlm) > 0:  # 使用 len() 函数检查数组是否为空
        for landmarks in landmarks_torchlm:
            landmarks_np = np.array(landmarks)  # 将 landmarks 转换为 NumPy 数组
            image = torchlm.utils.draw_landmarks(image, landmarks=landmarks_np, color=(255, 0, 0), thickness=2)  # 红色点和线表示 Torchlm 模型
    
    return image

class FaceProcessing(object):
    def __init__(self, ui_obj):
        self.name = "Face Image Processing"
        self.description = "Call for Face Image and video Processing"
        self.ui_obj = ui_obj

    def take_webcam_photo(self, image):
        return image

    def take_webcam_video(self, images):
        return images

    def mp_webcam_photo(self, image):
        return image

    def mp_webcam_face_mesh(self, image):
        mesh_image = apply_media_pipe_facemesh(image)
        return mesh_image

    def mp_webcam_face_detection(self, image):
        face_detection_img = apply_media_pipe_face_detection(image)
        return face_detection_img

    def dlib_apply_face_orientation(self, image):
        image = face_orientation_obj.create_orientation(image)
        return image

    def webcam_stream_update(self, video_frame):
        video_out = face_orientation_obj.create_orientation(video_frame)
        return video_out 
    
    def mp_webcam_face_68(self, image):
        face_68_img = apply_media_pipe_face_68(image)
        return face_68_img
    
    def tm_webcam_face_68(self, image):
        face_68_img = apply_torchlm_face_68(image)
        return face_68_img

    def compare_models(self, image):
        face_68_img = apply_line_of_corresponding_points(image)
        return face_68_img

    def create_ui(self):
        with self.ui_obj:
            gr.Markdown("Face Analysis with Webcam/Video")
            with gr.Tabs():
                with gr.TabItem("Playing with Webcam"):
                    with gr.Row():
                        webcam_image_in = gr.Image(label="Webcam Image Input" )
                        webcam_video_in = gr.Video(label="Webcam Video Input" )
                    with gr.Row():
                        webcam_photo_action = gr.Button("Take the Photo")
                        webcam_video_action = gr.Button("Take the Video")
                    with gr.Row():
                        webcam_photo_out = gr.Image(label="Webcam Photo Output")
                        webcam_video_out = gr.Video(label="Webcam Video")
                
                with gr.TabItem("Mediapipe Facemesh with Webcam"):
                    with gr.Row():
                        with gr.Column():
                            mp_image_in = gr.Image(label="Webcam Image Input" )
                        with gr.Column():
                            mp_photo_action = gr.Button("Take the Photo")
                            mp_apply_fm_action = gr.Button("Apply Face Mesh the Photo")
                            mp_apply_landmarks_action = gr.Button("Apply Face Landmarks the Photo")
                    with gr.Row():
                        mp_photo_out = gr.Image(label="Webcam Photo Output")
                        mp_fm_photo_out = gr.Image(label="Face Mesh Photo Output")
                        mp_lm_photo_out = gr.Image(label="Face Landmarks Photo Output")
                
                with gr.TabItem("DLib Based Face Orientation"):
                    with gr.Row():
                        with gr.Column():
                            dlib_image_in = gr.Image(label="Webcam Image Input" )
                        with gr.Column():
                            dlib_photo_action = gr.Button("Take the Photo")
                            dlib_apply_orientation_action = gr.Button("Apply Face Mesh the Photo")
                    with gr.Row():
                        dlib_photo_out = gr.Image(label="Webcam Photo Output")
                        dlib_orientation_photo_out = gr.Image(label="Face Mesh Photo Output")
                
                with gr.TabItem("Face Orientation on Live Webcam Stream"):
                    with gr.Row():
                        webcam_stream_in = gr.Image(label="Webcam Stream Input",
                                                    streaming=True)
                        webcam_stream_out = gr.Image(label="Webcam Stream Output")
                        webcam_stream_in.change(
                            self.webcam_stream_update,
                            inputs=webcam_stream_in,
                            outputs=webcam_stream_out
                        )
                with gr.TabItem("Models Comparison"):
                    with gr.Row():
                        with gr.Column():
                            original_video_in = gr.Image(label="Webcam Image Input" )
                        with gr.Column():
                            apply_landmarks_action = gr.Button("Apply Face Landmarks the Photo")
                            compare_models_action = gr.Button("Compare Models")
                    with gr.Row():
                        mp_68_points_image = gr.Image(label="MediaPipe 68 Points Video")
                        torchlm_68_points_image = gr.Image(label="Torchlm 68 Points Video")
                        comparison_image = gr.Image(label="Comparison")

            apply_landmarks_action.click(
                self.mp_webcam_face_68,
                [
                    original_video_in
                ],
                [
                    mp_68_points_image
                ]
            )

            apply_landmarks_action.click(
                self.tm_webcam_face_68,
                [
                    original_video_in
                ],
                [
                    torchlm_68_points_image
                ]
            )
            
            compare_models_action.click(
                self.compare_models,
                [
                    original_video_in
                ],
                [
                    comparison_image
                ]
            )

            dlib_photo_action.click(
                self.mp_webcam_photo,
                [
                    dlib_image_in
                ],
                [
                    dlib_photo_out
                ]
            )

            dlib_apply_orientation_action.click(
                self.dlib_apply_face_orientation,
                [
                    dlib_image_in
                ],
                [
                    dlib_orientation_photo_out
                ]
            )

            mp_photo_action.click(
                self.mp_webcam_photo,
                [
                    mp_image_in
                ],
                [
                    mp_photo_out
                ]
            )

            mp_apply_fm_action.click(
                self.mp_webcam_face_mesh,
                [
                    mp_image_in
                ],
                [
                    mp_fm_photo_out
                ]
            )

            mp_apply_landmarks_action.click(
                self.mp_webcam_face_detection,
                [
                    mp_image_in
                ],
                [
                    mp_lm_photo_out
                ]
            )

            webcam_photo_action.click(
                self.take_webcam_photo,
                [
                    webcam_image_in
                ],
                [
                    webcam_photo_out
                ]
            )

            webcam_video_action.click(
                self.take_webcam_video,
                [
                    webcam_video_in
                ],
                [
                    webcam_video_out
                ]
            )

    def launch_ui(self):
        self.ui_obj.launch(share=True)


if __name__ == '__main__':
    my_app = gr.Blocks()
    face_ui = FaceProcessing(my_app)
    face_ui.create_ui()
    face_ui.launch_ui()

