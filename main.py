import numpy as np
from PIL import ImageGrab
import cv2
import time
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
from processes_image import core

# Hàm lấy mức sử dụng GPU
def get_gpu_usage():
    try:
        handle = nvmlDeviceGetHandleByIndex(0)  # Lấy GPU đầu tiên
        utilization = nvmlDeviceGetUtilizationRates(handle)
        return utilization.gpu  # % sử dụng GPU
    except:
        return 0  # Nếu không có GPU, trả về 0

# Khởi tạo NVML
nvmlInit()

# Biến thời gian
last_time = time.time()
update_interval = 1  # Thời gian cập nhật (giây)
cpu_usage, gpu_usage = 0, 0  # Khởi tạo giá trị CPU và GPU

while True:
    # Lấy ảnh màn hình
    original_image = np.array(ImageGrab.grab(bbox=(0, 25, 800, 625)))

    points, vecto_img, ftime_ms = core.edge_detection_to_vector(original_image, 200, 300, 3, (3,3), 0)
    image = core.vector_to_image(points, original_image.shape)

    # Tính thời gian xử lý
    process_time = time.time() - last_time

    # Cập nhật CPU, GPU và thời gian sau mỗi giây
    if process_time >= update_interval:
        last_time = time.time()
        cpu_usage = psutil.cpu_percent(interval=0)  # % CPU
        gpu_usage = get_gpu_usage()  # % GPU
        print(f"\rLook took {process_time:.2f} seconds | CPU: {cpu_usage}% | GPU: {gpu_usage}% | Time image to vecto(ms): {ftime_ms} ms", end="")

    # Hiển thị ảnh
    cv2.imshow('window', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Dừng chương trình khi nhấn 'q'
    if cv2.waitKey(25) == ord('q'):
        cv2.destroyAllWindows()
        break
