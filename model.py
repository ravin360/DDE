from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loadDataset import *
from metrics import *

class DisparityCNN(nn.Module):
    def __init__(self):
        super(DisparityCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)  # 6 channels (left + right)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)  # Output 1 channel for disparity

    def forward(self, left_img, right_img):
        x = torch.cat((left_img, right_img), dim=1)  # Concatenate along the channel dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        disparity_map = self.output_conv(x)
        return disparity_map


class StereoDataset(Dataset):
    def __init__(self, left_images, right_images, m_values, target_size=(256, 512)):
        self.left_images = torch.tensor(left_images, dtype=torch.float32).permute(0, 3, 1, 2)
        self.right_images = torch.tensor(right_images, dtype=torch.float32).permute(0, 3, 1, 2)
        self.m_values = torch.tensor(m_values, dtype=torch.float32)
        self.m_values = self.m_values.unsqueeze(1)

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        left_img = self.left_images[idx]
        right_img = self.right_images[idx]
        m_val = self.m_values[idx]
        return left_img, right_img, m_val

dataset = StereoDataset(left_img, right_img, m)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
model = DisparityCNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for left_img, right_img, m_val in data_loader:
        left_img, right_img, m_val = left_img.to(device), right_img.to(device), m_val.to(device)

        optimizer.zero_grad()
        outputs = model(left_img, right_img)
        print("Output shape:", outputs.shape)
        print("m_val shape:", m_val.shape)
        loss = criterion(outputs, m_val)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("Training complete.")

model.eval()
for i, (left_image, right_image, m_val) in enumerate(zip(left_img, right_img, m)):
    with torch.no_grad():
        sample_left = left_image.float().unsqueeze(0).permute(0, 1, 2, 3).to(device)
        sample_right = right_image.float().unsqueeze(0).permute(0, 1, 2, 3).to(device)

        predicted_disparity = model(sample_left, sample_right)
        target_disparity = torch.tensor(m_val, dtype=torch.float32).unsqueeze(0).to(device)
        mae, rmse, bad_pixel_errors = compute_metrics(predicted_disparity, target_disparity)

        print(f"Image Pair {i + 1}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        for threshold, bad_percentage in bad_pixel_errors.items():
            print(f"{threshold}: {bad_percentage:.2f}%")

        focal_length = 721.5377
        baseline = 63.7

        # Convert the predicted disparity to a depth map
        depth_map = (focal_length * baseline) / (
                    predicted_disparity.squeeze().cpu().numpy() + 1e-6)  # Adding a small value to avoid division by zero

        # Display the depth map
        plt.figure(figsize=(8, 6))
        plt.imshow(depth_map, cmap='grey')
        plt.colorbar(label="Depth (meters)")
        plt.title("Predicted Depth Map")
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.imshow(predicted_disparity.squeeze().cpu().numpy(), cmap='plasma')
        plt.colorbar()
        plt.title(f"Predicted Disparity Map for Image Pair {i + 1}")
        plt.show()

        if i == 4:
            break
