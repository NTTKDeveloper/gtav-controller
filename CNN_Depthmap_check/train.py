import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from depthmap_model import DepthMapCNN  # Import lớp mô hình của bạn
from pytorch_msssim import ssim


# Dataset class
class DepthMapDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.noisy_images = sorted(os.listdir(noisy_dir))
        self.clean_images = sorted(os.listdir(clean_dir))
        self.transform = transform

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy_image_path = os.path.join(self.noisy_dir, self.noisy_images[idx])
        clean_image_path = os.path.join(self.clean_dir, self.clean_images[idx])

        noisy_image = Image.open(noisy_image_path).convert('L')
        clean_image = Image.open(clean_image_path).convert('L')

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        return noisy_image, clean_image

# Validation function
def validate_model(model, dataloader, criterion):
    model.eval()  # Đặt chế độ eval
    val_loss = 0.0
    with torch.no_grad():  # Tắt gradient để giảm bộ nhớ
        for noisy_images, clean_images in dataloader:
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            # Forward pass
            outputs = model(noisy_images)
            loss = criterion(outputs, clean_images)
            val_loss += loss.item()

    return val_loss / len(dataloader)  # Trả về loss trung bình



def gradient_loss(pred, target):
    def compute_gradient(img):
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
        return dx, dy

    pred_dx, pred_dy = compute_gradient(pred)
    target_dx, target_dy = compute_gradient(target)

    dx_loss = torch.mean(torch.abs(pred_dx - target_dx))
    dy_loss = torch.mean(torch.abs(pred_dy - target_dy))

    return dx_loss + dy_loss

# Kết hợp Gradient Loss với MSE và SSIM
def combination_loss_with_gradient(output, target, alpha=0.84, beta=0.1):
    mse_loss = nn.MSELoss()(output, target)  # MSE
    ssim_l = 1 - ssim(output, target, data_range=1.0, size_average=True)  # SSIM
    grad_loss = gradient_loss(output, target)  # Gradient Loss
    return mse_loss + alpha * ssim_l + beta * grad_loss  # Tổng hợp Loss

# Sửa hàm train_model để sử dụng combination_loss_with_gradient
def train_model(model, train_loader, val_loader, optimizer, num_epochs, model_path, patience=10, min_delta=1e-4):
    loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for noisy_images, clean_images in train_loader:
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            # Forward pass
            outputs = model(noisy_images)
            
            # Sử dụng Combination Loss
            loss = combination_loss_with_gradient(outputs, clean_images, alpha=0.84, beta=0.1)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate training loss
        train_loss = running_loss / len(train_loader)
        loss_history.append(train_loss)

        # Validate the model
        val_loss = validate_model(model, val_loader, lambda output, target: combination_loss_with_gradient(output, target, alpha=0.84, beta=0.1))
        val_loss_history.append(val_loss)

        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Check for improvement in validation loss
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Display loss history
    print("Training completed. Loss history:")
    for epoch, (train_loss, val_loss) in enumerate(zip(loss_history, val_loss_history), 1):
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

# Paths
data_dir = "data"
val_dir = "val"
noisy_dir = os.path.join(data_dir, "vision_depthmap")
clean_dir = os.path.join(data_dir, "truth")
val_noisy_dir = os.path.join(val_dir, "vision_depthmap")
val_clean_dir = os.path.join(val_dir, "truth")
model_dir = "model"
model_path = os.path.join(model_dir, "depthmap_cnn.pth")

# Ensure model directory exists
os.makedirs(model_dir, exist_ok=True)

size_input = 200

# Data transformations
transform = transforms.Compose([
    transforms.Resize((size_input, int(size_input * 3 / 4))),  # Resize về kích thước cố định
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Dataset and DataLoader
train_dataset = DepthMapDataset(noisy_dir, clean_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

val_dataset = DepthMapDataset(val_noisy_dir, val_clean_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, criterion, optimizer
model = DepthMapCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Load model if exists
if os.path.exists(model_path):
    print("Loading existing model...")
    model.load_state_dict(torch.load(model_path))

# Train the model
num_epochs = 50
train_model(model, train_loader, val_loader, optimizer, num_epochs, model_path)

print("Training completed.")
