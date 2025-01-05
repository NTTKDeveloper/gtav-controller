import os
import time
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from depthmap_model import DepthMapCNN  # Import lớp mô hình đã tạo

# Đường dẫn đến thư mục test
test_dir = "test"

# Kiểm tra nếu thư mục test tồn tại
if not os.path.exists(test_dir):
    raise ValueError(f"Test directory '{test_dir}' not found.")

# Thiết lập thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải mô hình
model = DepthMapCNN().to(device)
model_path = "model/depthmap_cnn.pth"  # Thay đổi nếu đường dẫn mô hình khác
if not os.path.exists(model_path):
    raise ValueError(f"Model file '{model_path}' not found.")
model.load_state_dict(torch.load(model_path))
model.eval()

size_input = 200

# Data transformations
transform = transforms.Compose([
    transforms.Resize((size_input, int(size_input * 3 / 4))),  # Resize về kích thước cố định
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Đọc danh sách file trong thư mục test
test_images = sorted(os.listdir(test_dir))
if not test_images:
    raise ValueError(f"No images found in directory '{test_dir}'.")

# Biến lưu thời gian xử lý
processing_times = []

# Kích thước mới cho ảnh output
output_size = (800, 600)  # Kích thước resize

# Xử lý từng ảnh
for img_name in test_images:
    img_path = os.path.join(test_dir, img_name)
    if not img_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
        print(f"Skipping non-image file: {img_name}")
        continue

    # Mở ảnh và áp dụng transform
    image = Image.open(img_path).convert('L')
    input_tensor = transform(image).unsqueeze(0).to(device)  # Thêm batch dimension

    # Đo thời gian xử lý
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    end_time = time.time()

    # Tính thời gian và lưu lại
    processing_time = end_time - start_time
    processing_times.append(processing_time)

    print(f"Processed {img_name}: {processing_time:.4f} seconds")

    # Chuyển đổi tensor đầu ra từ mô hình thành numpy array để hiển thị bằng OpenCV
    output_image = output.squeeze().cpu().numpy()
    output_image = (output_image * 255).astype(np.uint8)  # Chuyển đổi lại để hiển thị
    print(output_image.shape)

    # Resize ảnh output lên 800x600
    output_image_resized = cv2.resize(output_image, output_size)

    # Chuyển ảnh input thành numpy array (ảnh đầu vào)
    input_image = np.array(image)

    # Resize ảnh đầu vào lên 800x600 (nếu cần)
    input_image_resized = cv2.resize(input_image, output_size)

    # Hiển thị ảnh input và output bằng OpenCV
    cv2.imshow('Noisy Image', input_image_resized)  # Hiển thị ảnh noisy
    cv2.imshow('Processed Image', output_image_resized)  # Hiển thị ảnh đầu ra đã resize

    # Đợi cho đến khi nhấn phím bất kỳ để đóng cửa sổ ảnh
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Tính tốc độ trung bình
average_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
print(f"\nAverage processing time per image: {average_time:.4f} seconds")
