import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 1. 定義與 BCPolicy 相同的特徵提取器
class BCPolicyFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2), nn.ReLU(),
            nn.Flatten()
        )
        # 計算 flatten 後的維度
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            n_flat = self.cnn(dummy).shape[1]
        self.n_flat = n_flat
        self.fc = nn.Sequential(
            nn.Linear(n_flat + 1, 128), nn.ReLU()
        )

    def forward(self, obs):
        # obs: dict with 'segmentation' and 'speed'
        img = obs["segmentation"]
        speed = obs["speed"]
        x = self.cnn(img)
        x = torch.cat([x, speed], dim=1)
        return self.fc(x)

# 2. 定義 policy_kwargs
policy_kwargs = dict(
    features_extractor_class=BCPolicyFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=[dict(pi=[64], vf=[64])]
)

# 3. 定義 BCPolicy
class BCPolicy(nn.Module):
    def __init__(self, img_size=(3, 64, 64)):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2), nn.ReLU(),
            nn.Flatten()
        )
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
