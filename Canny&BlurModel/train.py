import torch
import torch.nn as nn
import torch.optim as optim

# Định nghĩa mô hình
class ImageProcessingModel(nn.Module):
    def __init__(self, input_size):
        super(ImageProcessingModel, self).__init__()
        
        # Các lớp fully connected chung
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # Đầu ra cho kmedian_blurred (số nguyên dương lẻ: 3, 5, 7,...)
        self.fc_kmedian = nn.Linear(64, 3)  # Dự đoán các giá trị 3, 5, 7
        
        # Đầu ra cho kgaussian_blurred (width, height - cặp số nguyên dương lẻ)
        self.fc_kgaussian_width = nn.Linear(64, 1)
        self.fc_kgaussian_height = nn.Linear(64, 1)
        
        # Đầu ra cho sigma (số thực dương)
        self.fc_sigma = nn.Linear(64, 1)
        
        # Đầu ra cho threshold1 và threshold2 (số thực dương)
        self.fc_threshold1 = nn.Linear(64, 1)
        self.fc_threshold2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # Các lớp trung gian
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Dự đoán kmedian_blurred
        kmedian_blurred = torch.softmax(self.fc_kmedian(x), dim=1)  # Softmax để chọn một trong các giá trị lẻ
        
        # Dự đoán kgaussian_blurred (width, height)
        kgaussian_width = torch.softmax(self.fc_kgaussian_width(x), dim=1)
        kgaussian_height = torch.softmax(self.fc_kgaussian_height(x), dim=1)
        
        # Dự đoán sigma
        sigma = self.fc_sigma(x)
        
        # Dự đoán threshold1 và threshold2
        threshold1 = self.fc_threshold1(x)
        threshold2 = self.fc_threshold2(x)
        
        return kmedian_blurred, kgaussian_width, kgaussian_height, sigma, threshold1, threshold2

# Hàm huấn luyện mô hình
def train_model(model, X_train, y_kmedian, y_kgaussian, y_sigma, y_threshold, epochs=100, batch_size=32):
    # Chuyển dữ liệu sang tensor PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_kmedian_tensor = torch.tensor(y_kmedian, dtype=torch.long)
    y_kgaussian_tensor = torch.tensor(y_kgaussian, dtype=torch.float32)
    y_sigma_tensor = torch.tensor(y_sigma, dtype=torch.float32)
    y_threshold_tensor = torch.tensor(y_threshold, dtype=torch.float32)
    
    # Chọn hàm mất mát và bộ tối ưu
    criterion_kmedian = nn.CrossEntropyLoss()  # Dành cho phân loại (kmedian)
    criterion_kgaussian = nn.MSELoss()  # Dành cho kgaussian
    criterion_sigma = nn.MSELoss()  # Dành cho sigma
    criterion_threshold = nn.MSELoss()  # Dành cho threshold1 và threshold2
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Huấn luyện mô hình
    for epoch in range(epochs):
        model.train()  # Đặt mô hình ở chế độ huấn luyện
        optimizer.zero_grad()  # Làm sạch gradient trước khi tính toán mới
        
        # Forward pass
        kmedian_blurred, kgaussian_width, kgaussian_height, sigma, threshold1, threshold2 = model(X_train_tensor)
        
        # Tính toán mất mát
        loss_kmedian = criterion_kmedian(kmedian_blurred, y_kmedian_tensor)
        loss_kgaussian = criterion_kgaussian(kgaussian_width, y_kgaussian[:, 0]) + criterion_kgaussian(kgaussian_height, y_kgaussian[:, 1])
        loss_sigma = criterion_sigma(sigma.squeeze(), y_sigma_tensor)
        loss_threshold1 = criterion_threshold(threshold1.squeeze(), y_threshold_tensor[:, 0])
        loss_threshold2 = criterion_threshold(threshold2.squeeze(), y_threshold_tensor[:, 1])
        
        # Tổng hợp mất mát
        total_loss = loss_kmedian + loss_kgaussian + loss_sigma + loss_threshold1 + loss_threshold2
        
        # Backward pass và tối ưu hóa
        total_loss.backward()
        optimizer.step()
        
        # In ra mất mát mỗi epoch
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}')
    
    return model

# Giả sử bạn đã có dữ liệu huấn luyện X_train, y_kmedian, y_kgaussian, y_sigma, y_threshold
# Ví dụ:
# X_train: Dữ liệu đầu vào (có thể là ảnh hoặc đặc trưng đã trích xuất)
# y_kmedian: Nhãn cho kmedian_blurred (ví dụ: 3, 5, 7)
# y_kgaussian: Nhãn cho kgaussian_blurred (ví dụ: (5, 5), (3, 3))
# y_sigma: Nhãn cho sigma
# y_threshold: Nhãn cho threshold1 và threshold2

# Khởi tạo mô hình
input_size = 10  # Số lượng đặc trưng đầu vào
model = ImageProcessingModel(input_size)

# Train mô hình
# model = train_model(model, X_train, y_kmedian, y_kgaussian, y_sigma, y_threshold)
