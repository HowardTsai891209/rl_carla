import carla
import pygame
import random
import time
import numpy as np
import os
import csv

# CARLA 連線參數
HOST = '127.0.0.1'
PORT = 2000
IMG_WIDTH = 128
IMG_HEIGHT = 128

# 初始化 pygame
pygame.init()
screen = pygame.display.set_mode((IMG_WIDTH, IMG_HEIGHT))
pygame.display.set_caption('CARLA Autopilot Toggle Demo (press P to toggle)')
clock = pygame.time.Clock()

# 連線到 CARLA
client = carla.Client(HOST, PORT)
client.set_timeout(10.0)
world = client.get_world()

# Spawn vehicle
blueprint_library = world.get_blueprint_library()
vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if vehicle is None:
    raise RuntimeError('Failed to spawn vehicle!')

# Spawn camera sensor
cam_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
cam_bp.set_attribute('image_size_x', str(IMG_WIDTH))
cam_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
cam_bp.set_attribute('fov', '90')
cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

# 影像緩衝與資料收集
image_surface = None
last_obs = None
expert_data = []  # 收集 (image, speed, steer, throttle, brake)

# 建立資料夾
os.makedirs('images', exist_ok=True)
label_file = open('expert_labels.csv', 'w', newline='')
label_writer = csv.writer(label_file)
label_writer.writerow(['filename', 'speed', 'steer', 'throttle', 'brake'])

frame_idx = 0

running = True

def process_img(image):
    global image_surface, last_obs, running
    if not running:
        return
    image.convert(carla.ColorConverter.CityScapesPalette)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((IMG_HEIGHT, IMG_WIDTH, 4))[:, :, :3]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    last_obs = array  # 存下來
camera.listen(process_img)

# 初始自動駕駛狀態
autopilot = True
vehicle.set_autopilot(autopilot)

running = True
while running:
    # 檢查 vehicle/camera 是否還活著
    if not (vehicle.is_alive and camera.is_alive):
        print("Actor destroyed, exiting...")
        running = False
        break
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                autopilot = not autopilot
                vehicle.set_autopilot(autopilot)
                print(f"Autopilot: {'ON' if autopilot else 'OFF'}")
    # 收集資料（僅 autopilot 狀態下）
    if autopilot and image_surface is not None and last_obs is not None:
        control = vehicle.get_control()
        v = vehicle.get_velocity()
        speed = 3.6 * (v.x**2 + v.y**2 + v.z**2)
        # 收集全速域資料（只排除完全靜止）
        if speed > 1:
            img_name = f"images/img_{frame_idx:06d}.png"
            # 儲存影像
            from PIL import Image
            Image.fromarray(last_obs).save(img_name)
            # 寫入標籤
            label_writer.writerow([img_name, speed, control.steer, control.throttle, control.brake])
            frame_idx += 1
            # 收集到 25000 筆自動結束
            if frame_idx >= 25000:
                print("Collected 25000 samples, exiting...")
                running = False
                break
    # 顯示影像
    if image_surface is not None:
        screen.blit(image_surface, (0, 0))
    pygame.display.flip()
    clock.tick(30)

# 清理
running = False  # <--- 先設為 False，讓 callback 停止處理
camera.stop()
camera.destroy()
vehicle.destroy()
pygame.quit()
label_file.close()
print(f"Saved {frame_idx} expert samples to images/ and expert_labels.csv")
