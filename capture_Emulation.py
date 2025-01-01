import cv2
import time
import numpy as np

def compute_depth_map(disparity_map, fov, baseline_cm, width_px):
    fov_rad = np.deg2rad(fov)
    f = width_px / (2 * np.tan(fov_rad / 2))
    baseline_m = baseline_cm / 100.0
    depth_map = np.zeros_like(disparity_map, dtype=np.float32)
    disparity_map = disparity_map.astype(np.float32)
    non_zero_disparity = disparity_map > 0
    depth_map[non_zero_disparity] = (f * baseline_m) / disparity_map[non_zero_disparity]
    return depth_map

cap = cv2.VideoCapture("EmulationMSAA.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = 1000 / fps

# Tạo bộ tính disparity map
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=9)

print(f"FPS: {fps}")

while True:
    start_time = time.time()
    ret, img = cap.read()
    if not ret:
        break

    # img = cv2.medianBlur(img, 7)

    # Chia ảnh thành hai nửa
    h, w, _ = img.shape
    # print(img.shape)
    
    left = img[:, :w // 2, :]
    right = img[:, w // 2:, :]

    left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    #Làm mờ 
    right = cv2.medianBlur(right, 5)
    left = cv2.medianBlur(left, 5)
    left = cv2.GaussianBlur(left, (5, 5), 0)
    right = cv2.GaussianBlur(right, (5, 5), 0)

    threshold2 = 100
    threshold1 = 10
    # left = cv2.Canny(left, threshold1=threshold1, threshold2=threshold2)
    # right = cv2.Canny(right, threshold1=threshold1, threshold2=threshold2)
    

    # Tính disparity map
    disparity_map = stereo.compute(left, right).astype(np.float32) / 16.0
    disparity_map = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    disparity_map = cv2.medianBlur(disparity_map, 5)

    # Tính depth map
    depth_map = compute_depth_map(disparity_map, fov=90, baseline_cm=6, width_px=left.shape[1])

    # Hiển thị kết quả
    cv2.imshow("Vision", img)
    cv2.imshow("Left", left)
    cv2.imshow("Right", right)
    cv2.imshow("Disparity_map", disparity_map)
    cv2.imshow("Depth_map", depth_map / np.max(depth_map) * 255)

    # Điều chỉnh thời gian chờ
    elapsed_time = (time.time() - start_time) * 1000  # Thời gian xử lý
    delay_time = max(int(wait_time - elapsed_time), 1)
    if cv2.waitKey(delay_time) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
