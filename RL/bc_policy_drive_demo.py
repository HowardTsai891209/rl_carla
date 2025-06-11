import carla
import pygame
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# 1. 定義 BCPolicy（與 bc_train.py 完全一致）
class BCPolicy(nn.Module):
    def __init__(self, img_size=(3, 128, 128)):
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

# 2. 載入 BCPolicy 權重
policy = BCPolicy(img_size=(3, 128, 128))
policy.load_state_dict(torch.load("bc_policy.pth", map_location="cpu"))
policy.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy.to(device)

# 3. CARLA 參數
HOST = '127.0.0.1'
PORT = 2000
IMG_WIDTH = 336
IMG_HEIGHT = 336
MODEL_IMG_SIZE = 128

# 4. 初始化 pygame
pygame.init()
screen = pygame.display.set_mode((IMG_WIDTH, IMG_HEIGHT))
pygame.display.set_caption('BCPolicy Drive Demo (ESC to quit)')
clock = pygame.time.Clock()

# 5. 連線到 CARLA
client = carla.Client(HOST, PORT)
client.set_timeout(10.0)
world = client.get_world()

# 6. Spawn vehicle
blueprint_library = world.get_blueprint_library()
vehicle_bp = np.random.choice(blueprint_library.filter('vehicle.*'))
spawn_point = np.random.choice(world.get_map().get_spawn_points())
spawn_point.location.z += 1.5  # 提高 z 避免落地穿模
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if vehicle is None:
    raise RuntimeError('Failed to spawn vehicle!')

# 7. Spawn camera sensor
cam_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
cam_bp.set_attribute('image_size_x', str(IMG_WIDTH))
cam_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
cam_bp.set_attribute('fov', '90')
cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

# 8. 影像緩衝
image_surface = None
last_obs = None

def process_img(image):
    global image_surface, last_obs
    image.convert(carla.ColorConverter.CityScapesPalette)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((IMG_HEIGHT, IMG_WIDTH, 4))[:, :, :3]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    last_obs = array
camera.listen(process_img)

# 9. 控制主迴圈
stuck_counter = 0
stuck_threshold = 350  # 連續 5 秒 (假設 70Hz) 卡住就 respawn
min_speed = 1.0  # km/h
min_throttle = 0.3

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
    if image_surface is not None and last_obs is not None:
        # 取得速度
        v = vehicle.get_velocity()
        speed = 3.6 * (v.x**2 + v.y**2 + v.z**2)
        # 預處理影像（縮放到模型輸入大小）
        img_pil = Image.fromarray(last_obs)
        img_pil = img_pil.resize((MODEL_IMG_SIZE, MODEL_IMG_SIZE), Image.BILINEAR)
        img = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        spd = torch.tensor([[speed]], dtype=torch.float32)
        img = img.to(device)
        spd = spd.to(device)
        # BCPolicy 預測
        with torch.no_grad():
            action = policy(img, spd).cpu().numpy()[0]
        steer, throttle, brake = float(action[0]), float(action[1]), float(action[2])
        # 控制車輛
        control = carla.VehicleControl()
        control.steer = np.clip(steer, -1, 1)
        control.throttle = np.clip(throttle, 0, 1)
        control.brake = np.clip(brake, 0, 1)
        vehicle.apply_control(control)
        # 卡住偵測：油門大於 min_throttle 但速度小於 min_speed
        if throttle > min_throttle and speed < min_speed:
            stuck_counter += 1
        else:
            stuck_counter = 0
        if stuck_counter > stuck_threshold:
            print("[Info] Vehicle stuck, respawning at a new location...")
            new_spawn_point = np.random.choice(world.get_map().get_spawn_points())
            new_spawn_point.location.z += 1.5
            vehicle.set_transform(new_spawn_point)
            vehicle.set_target_velocity(carla.Vector3D(0,0,0))
            vehicle.set_target_angular_velocity(carla.Vector3D(0,0,0))
            stuck_counter = 0
        # 顯示畫面
        screen.blit(image_surface, (0, 0))
        # 顯示 speed, throttle, steer, brake, fps（左上角一行一個數值）
        font_size = max(16, IMG_HEIGHT // 12)
        font = pygame.font.SysFont(None, font_size)
        fps = clock.get_fps()
        info_lines = [
            f"Speed: {speed:.1f} km/h",
            f"Throttle: {throttle:.2f}",
            f"Steer: {steer:.2f}",
            f"Brake: {brake:.2f}",
            f"FPS: {fps:.1f}"
        ]
        for i, line in enumerate(info_lines):
            text_surface = font.render(line, True, (255, 255, 255))
            screen.blit(text_surface, (10, 10 + i * (font_size + 2)))
    pygame.display.flip()
    clock.tick(0)

# 10. 清理
camera.stop()
camera.destroy()
vehicle.destroy()
pygame.quit()
print('BCPolicy demo finished.')
