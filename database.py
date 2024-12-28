import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm  # Thư viện để hiển thị thanh tiến trình

# Hàm tạo ảnh và vẽ biên sử dụng OpenCV
def create_circle_with_smooth_edges(size, color1, color2, radius1, radius2):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    edges = np.zeros((size, size), dtype=np.uint8)

    center = (size // 2, size // 2)

    # Vẽ đường tròn bên ngoài
    cv2.circle(image, center, radius1, color1, -1)
    cv2.circle(edges, center, radius1, 255, 1)  # Vẽ biên ngoài

    # Vẽ đường tròn bên trong
    cv2.circle(image, center, radius2, color2, -1)
    cv2.circle(edges, center, radius2, 255, 1)  # Vẽ biên trong

    return image, edges

# Hàm random màu sắc gần giống nhau
def generate_similar_colors():
    base_color = [random.randint(50, 200) for _ in range(3)]  # Màu cơ bản (trung bình)
    delta = random.randint(5, 20)  # Độ chênh lệch nhỏ giữa các màu
    color1 = tuple(base_color)
    color2 = tuple(max(0, min(255, c - delta)) for c in base_color)  # Giảm giá trị mỗi kênh RGB
    return color1, color2

# Hàm tạo và lưu nhiều ảnh
def generate_and_save_images(num_images, size, outer_radius, inner_radius):
    # Thư mục lưu trữ
    data_dir = "data"
    label_dir = "label"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for i in tqdm(range(num_images), desc="Đang tạo ảnh"):  # Sử dụng tqdm để hiển thị thanh tiến trình
        # Random màu sắc
        outer_color, inner_color = generate_similar_colors()

        # Tạo ảnh và biên
        circle_image, edges_image = create_circle_with_smooth_edges(
            size, outer_color, inner_color, outer_radius, inner_radius
        )

        # Lưu ảnh
        image_filename = os.path.join(data_dir, f"circle_image_{i+1}.png")
        edges_filename = os.path.join(label_dir, f"edges_image_{i+1}.png")
        cv2.imwrite(image_filename, circle_image)
        cv2.imwrite(edges_filename, edges_image)

# Thông số
image_size = 100
outer_radius = 40
inner_radius = 20
num_images = 1000000 # Số lượng ảnh cần tạo

# Tạo và lưu ảnh
generate_and_save_images(num_images, image_size, outer_radius, inner_radius)

# Hiển thị ví dụ ảnh đầu tiên
example_image = cv2.imread(os.path.join("data", "circle_image_1.png"))
example_edges = cv2.imread(os.path.join("label", "edges_image_1.png"), cv2.IMREAD_GRAYSCALE)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Ảnh gốc (ví dụ)")
axes[0].axis("off")

axes[1].imshow(example_edges, cmap='gray')
axes[1].set_title("Ảnh biên (ví dụ)")
axes[1].axis("off")

plt.tight_layout()
plt.show()
