import cv2
import time
import numpy as np
from GyroController import GyroController as GC

def compute_depth_map(disparity_map, fov, baseline_cm, width_px):
    fov_rad = np.deg2rad(fov)
    f = width_px / (2 * np.tan(fov_rad / 2))
    baseline_m = baseline_cm / 100.0
    depth_map = np.zeros_like(disparity_map, dtype=np.float32)
    disparity_map = disparity_map.astype(np.float32)
    non_zero_disparity = disparity_map > 0
    depth_map[non_zero_disparity] = (f * baseline_m) / disparity_map[non_zero_disparity]
    return depth_map

def filter_gray_color(image):
    # Chuyển đổi ảnh sang không gian màu HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Định nghĩa phạm vi màu xám trong không gian HSV
    # Màu xám có Hue gần như bằng 0 (vì không có màu sắc), và Saturation rất thấp
    lower_gray = np.array([0, 0, 0])   # Giới hạn dưới: Hue=0, Sat=0, Value=50
    upper_gray = np.array([180, 30, 200])  # Giới hạn trên: Hue=180, Sat=30, Value=200
    
    # Tạo mặt nạ cho các vùng màu xám
    mask = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # Áp dụng mặt nạ lên ảnh gốc
    gray_filtered = cv2.bitwise_and(image, image, mask=mask)
    
    return gray_filtered

# Tạo bộ tính disparity map
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=5)

def Preprocessing_Vision(vision):
        # Chia ảnh thành hai nửa
    h, w, _ = vision.shape
    # print(img.shape)
    
    left = vision[:, :w // 2, :]
    right = vision[:, w // 2:, :]    
    
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)


    # Làm mờ 
    kBlur = 11
    right_Blur = cv2.medianBlur(right_gray, kBlur)
    left_Blur = cv2.medianBlur(left_gray, kBlur)
    right_Blur = cv2.GaussianBlur(right_Blur, (kBlur, kBlur), 0)
    left_Blur = cv2.GaussianBlur(left_Blur, (kBlur, kBlur), 0)

    # Tính disparity map
    disparity_map = stereo.compute(left_Blur, right_Blur).astype(np.float32) / 16.0
    disparity_map = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    disparity_map = cv2.medianBlur(disparity_map, 5)

    # Tính depth map
    depth_map = compute_depth_map(disparity_map, fov=90, baseline_cm=6.199993, width_px=left.shape[1])
    # Chuẩn hóa về 0-255
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype(np.uint8)

    # Chuyển từ grayscale sang BGR
    depth_normalized_bgr = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
    # print(depth_normalized_bgr.shape)

    # Lọc vùng màu xám
    depth_normalized_fgray = filter_gray_color(depth_normalized_bgr)

    # # Làm mờ 
    # kBlur = 11
    # depth_normalized_bgr = cv2.medianBlur(depth_normalized_bgr, kBlur)
    # depth_normalized_bgr = cv2.GaussianBlur(depth_normalized_bgr, (kBlur, kBlur), 0)

    # Kiểm tra nếu ảnh đã có 3 kênh màu (ảnh màu) và chuyển sang ảnh đơn kênh (grayscale)
    if len(depth_normalized_fgray.shape) == 3:
        binary_image = cv2.cvtColor(depth_normalized_fgray, cv2.COLOR_BGR2GRAY)
    
    # Tiền xử lý: Áp dụng phép co (Erosion) để làm mỏng đường viền
    kernel = np.ones((11,11), np.uint8)  # Kernel 3x3 để làm mỏng
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)

    # Phát hiện đường viền
    contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tạo ảnh trống để vẽ
    emulation_vision = np.zeros_like(binary_image)

    # Vẽ lại các đường viền trên ảnh trống
    cv2.drawContours(emulation_vision, contours, -1, (255), thickness=-1)

    # Hiển thị kết quả
    # cv2.imshow("Vision", vision)
    # cv2.imshow("Left", left)
    # cv2.imshow("Right", right)
    # cv2.imshow("Disparity_map", disparity_map)
    # cv2.imshow("Depth_map", depth_normalized)
    # cv2.imshow("Depth_normalized_gray", depth_normalized_fgray)
    # cv2.imshow("Emulation Vision", emulation_vision)
    # cv2.imshow("Erosion Image", eroded_image)
    return left, right, disparity_map, depth_normalized, depth_normalized_fgray, eroded_image, emulation_vision 

