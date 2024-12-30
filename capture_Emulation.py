import struct
from multiprocessing import shared_memory
import cv2
import numpy as np

def read_shared_memory():
    # Kết nối tới shared memory
    shm = shared_memory.SharedMemory(name="SharedMemory")
    buffer = shm.buf

    while True:
        # Đọc độ dài dữ liệu
        length = struct.unpack('I', buffer[:4])[0]

        if length > 0:
            # Đọc dữ liệu ảnh
            image_bytes = bytes(buffer[4:4 + length])

            # Chuyển đổi dữ liệu ảnh PNG thành numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)

            # Decode dữ liệu ảnh PNG thành định dạng OpenCV (BGR)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Hiển thị ảnh
            if image is not None:
                cv2.imshow("Shared Memory Image", image)

            # Nhấn phím 'q' để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Dọn dẹp tài nguyên
    shm.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    read_shared_memory()
