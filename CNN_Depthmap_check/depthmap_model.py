import torch.nn as nn

class DepthMapCNN(nn.Module):
    def __init__(self):
        super(DepthMapCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(alpha=1.0),  # Thay ReLU bằng ELU
            nn.Dropout(0.3),  # Tỷ lệ dropout
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(alpha=1.0),  # Thay ReLU bằng ELU
            nn.MaxPool2d(2, 2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ELU(alpha=1.0),  # Thay ReLU bằng ELU
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        skip = x  # lưu đầu vào ban đầu
        x = self.encoder(x)
        x = self.decoder(x)
        return x + skip  # cộng đầu vào với đầu ra để giữ thông tin gốc
