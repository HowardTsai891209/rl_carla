import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import numpy as np

# 1. Dataset
class CarlaBCDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        img_path = self.labels.iloc[idx, 0]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # 修正: 明確轉換為 float32，避免 object_ 型態
        speed = torch.tensor([float(self.labels.iloc[idx, 1])], dtype=torch.float32)
        action = torch.tensor([
            float(self.labels.iloc[idx, 2]),
            float(self.labels.iloc[idx, 3]),
            float(self.labels.iloc[idx, 4])
        ], dtype=torch.float32)
        return {'image': image, 'speed': speed, 'action': action}

# 2. Model
class BCPolicy(nn.Module):
    def __init__(self, img_size=(3, 128, 128)):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2), nn.ReLU(),
            nn.Flatten()
        )
        # 計算flatten後的維度
        with torch.no_grad():
            dummy = torch.zeros(1, *img_size)
            n_flat = self.cnn(dummy).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(n_flat + 1, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )
    def forward(self, image, speed):
        x = self.cnn(image)
        x = torch.cat([x, speed], dim=1)
        return self.fc(x)

if __name__ == '__main__':
    # 3. DataLoader
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = CarlaBCDataset('expert_labels.csv', 'images', transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    # 4. Model, Optimizer, Loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BCPolicy(img_size=(3, 128, 128)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # 速度分布統計（訓練前只做一次）
    df = pd.read_csv('expert_labels.csv')
    bins = np.arange(0, 310, 10)
    df['speed_bin'] = pd.cut(df['speed'], bins=bins, labels=False, right=False)
    bin_counts = df['speed_bin'].value_counts().sort_index().to_dict()
    max_count = max(bin_counts.values())

    # 5. Training loop
    for epoch in range(25):
        model.train()
        total_loss = 0
        for batch in loader:
            img = batch['image'].to(device)
            speed = batch['speed'].to(device)
            action = batch['action'].to(device)
            pred = model(img, speed)
            # 權重設計：
            # 1. 低速(0~19)給極低權重
            # 2. 高速(>200)給較低權重（因為高速直線多，且容易失控）
            # 3. 轉彎(steer絕對值>0.1)給高權重
            bin_idx = (speed.squeeze(1) // 10).long().cpu().numpy()
            steer_abs = action[:,0].abs().cpu().numpy()
            weights = []
            for i, b in enumerate(bin_idx):
                w = 1.0
                if b <= 1:  # 0~19 km/h
                    w = 0.05
                elif b >= 20:  # >200 km/h
                    w = 0.3
                else:
                    w = max_count / bin_counts.get(b, 1)
                # 轉彎(steer>0.1)再額外加權
                if steer_abs[i] > 0.1:
                    w *= 2.0
                weights.append(w)
            weights = torch.tensor(weights, dtype=torch.float32, device=device)
            loss = ((pred - action) ** 2).mean(dim=1) * weights
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * img.size(0)
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataset):.4f}')
    # 6. Save model (PyTorch)
    torch.save(model.state_dict(), 'bc_policy.pth')
    print('BC policy saved as bc_policy.pth')
    # 匯出 ONNX 以利轉 Keras
    dummy_img = torch.zeros(1, 3, 128, 128).to(device)
    dummy_speed = torch.zeros(1, 1).to(device)
    torch.onnx.export(model, (dummy_img, dummy_speed), "model_saved_from_CNN.onnx",
                      input_names=['image', 'speed'], output_names=['output'],
                      dynamic_axes={'image': {0: 'batch'}, 'speed': {0: 'batch'}, 'output': {0: 'batch'}})
    print('Model exported to model_saved_from_CNN.onnx')
    # ONNX 轉 Keras 並存成 H5
    try:
        from onnx2keras import onnx_to_keras
        import onnx
        import tensorflow as tf
        onnx_model = onnx.load('model_saved_from_CNN.onnx')
        k_model = onnx_to_keras(onnx_model, ['image', 'speed'])
        k_model.save('model_saved_from_CNN.h5')
        print('Keras model saved as model_saved_from_CNN.h5')
    except Exception as e:
        print('ONNX to Keras/H5 conversion failed:', e)

