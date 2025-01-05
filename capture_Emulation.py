import cv2
import time
import numpy as np
from GyroController import GyroController as GC
import pre_processing

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

cap = cv2.VideoCapture("./video/data.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = 1000 / fps

# Tạo bộ tính disparity map
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=5)

print(f"FPS: {fps}")

# Đường dẫn để lưu hình ảnh
output_path = "./CNN_Depthmap_check/data/"
count_img = 0
last_time_chip = time.time()
update_interval = 1  # Thời gian cập nhật (giây)

while True:
    start_time = time.time()
    ret, img = cap.read()
    if not ret:
        break

    #Tiền xử lí ảnh
    left, right, _, depth_normalized, depth = pre_processing.preprocessing_vision(img)

    
    process_time_chip = time.time() - last_time_chip
    # Lấy giá trị chipset sau update_interval
    if process_time_chip >= update_interval:
        # Lưu hình ảnh
        print("Đang lưu ảnh")
        cv2.imwrite(f"{output_path}/vision_depthmap/" + f"{count_img}.png", depth_normalized)
        cv2.imwrite(f"{output_path}/truth/" + f"{count_img}.png", depth)
        count_img = count_img + 1


    # print(f"Hình ảnh đã được lưu vào: {output_path}")

    # Hiển thị ảnh
    # cv2.imshow('Vision', cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    cv2.imshow('Vision depth_normalized', cv2.cvtColor(depth_normalized, cv2.COLOR_BGR2RGB))
    cv2.imshow('Vision depth', cv2.cvtColor(depth, cv2.COLOR_BGR2RGB))

    # Điều chỉnh thời gian chờ
    elapsed_time = (time.time() - start_time) * 1000  # Thời gian xử lý
    delay_time = max(int(wait_time - elapsed_time), 1)
    if cv2.waitKey(delay_time) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
