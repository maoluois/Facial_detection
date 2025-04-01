import cv2
import numpy as np

# 设置棋盘格的内角点数量
chessboard_size = (8, 5)
square_size = 15.0  # 以毫米为单位

# 准备棋盘格的3D点
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 存储3D点和2D点
objpoints = []
imgpoints = []

# 打开视频流
cap = cv2.VideoCapture(0)  # 0 表示默认相机

print("按 'c' 键拍摄照片进行标定，按 'q' 键退出。")

captured_images = 0
total_images = 60

while captured_images < total_images:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, 
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                             cv2.CALIB_CB_NORMALIZE_IMAGE + 
                                             cv2.CALIB_CB_FAST_CHECK)

    if ret:
        # 提高角点精度
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # 可视化角点
        cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret)

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') and ret:
        # 拍摄照片并保存角点
        objpoints.append(objp)
        imgpoints.append(corners2)
        captured_images += 1
        print(f"照片已拍摄并保存角点 ({captured_images}/{total_images})。")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 进行相机标定
if len(objpoints) > 0 and len(imgpoints) > 0:
    print("开始相机标定...")
    print(f"objpoints 数量: {len(objpoints)}")
    print(f"imgpoints 数量: {len(imgpoints)}")
    print(f"图像尺寸: {gray.shape[::-1]}")

    try:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("相机标定完成。")

        # 打印相机内参矩阵
        print("Camera matrix:")
        print(camera_matrix)

        # 打印畸变系数
        print("Distortion coefficients:")
        print(dist_coeffs)
    except Exception as e:
        print(f"相机标定失败: {e}")
else:
    print("未找到足够的棋盘格角点进行标定。")