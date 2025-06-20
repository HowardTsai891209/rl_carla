'''
building new env:

Steering - continous - done
Acceleration - smooth continous
Clean up at completion of episode
Navigation - define a goal to travel a short distance

'''

import random
import time
import numpy as np
import math 
import cv2
import gym
from gym import spaces
import carla
from tensorflow.keras.models import load_model
import os
import queue

SECONDS_PER_EPISODE = 15

N_CHANNELS = 3
HEIGHT = 240
WIDTH = 320

SPIN = 8

HEIGHT_REQUIRED_PORTION = 0.5 #bottom share, e.g. 0.1 is take lowest 10% of rows
WIDTH_REQUIRED_PORTION = 0.9

FIXED_DELTA_SECONDS = 0.3

SHOW_PREVIEW = False # 修改此處以關閉畫面顯示

class CarEnv(gym.Env):
	SHOW_CAM = SHOW_PREVIEW
	STEER_AMT = 1.0
	im_width = WIDTH
	im_height = HEIGHT
	front_camera = None
	CAMERA_POS_Z = 1.3 
	CAMERA_POS_X = 1.4
	PREFERRED_SPEED = 10 # what it says
	SPEED_THRESHOLD = 2 #defines when we get close to desired speed so we drop the
	
	def __init__(self):
		super(CarEnv, self).__init__()
		self.actor_list = [] # Initialize actor_list here
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using continous actions:
		# self.action_space = spaces.Box(low=-1, high=1,shape=(2,),dtype=np.uint8)
		# now we use descrete actions
		# self.action_space = spaces.MultiDiscrete([9])
		self.action_space = spaces.MultiDiscrete([9])
        # discrete variable with 9 possible actions for steering with middle being straight
        # REMOVED Second discrete variable with 4 possible actions for throttle/braking - removed
		self.height_from = int(HEIGHT * (1 -HEIGHT_REQUIRED_PORTION))
		self.width_from = int((WIDTH - WIDTH * WIDTH_REQUIRED_PORTION) / 2)
		self.width_to = self.width_from + int(WIDTH_REQUIRED_PORTION * WIDTH)
		self.new_height = HEIGHT - self.height_from
		self.new_width = self.width_to - self.width_from
		self.image_for_CNN = None
        # Example for using image as input normalised to 0..1 (channel-first; channel-last also works):
		self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(7, 18, 8), dtype=np.float32)
		self.client = carla.Client("localhost", 2000)
		self.client.set_timeout(4.0)
		self.world = self.client.get_world()
		self.settings = self.world.get_settings()
		self.settings.no_rendering_mode = True
		self.settings.synchronous_mode = True  # 改為同步模式
		self.settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
		self.world.apply_settings(self.settings)
		self.blueprint_library = self.world.get_blueprint_library()
		self.model_3 = self.blueprint_library.filter("model3")[0]
		cnn_path = os.path.join(os.path.dirname(__file__), 'model_saved_from_CNN.h5')
		self.cnn_model = load_model(cnn_path, compile=False) # 使用相對路徑
		self.cnn_model.compile()
		self._last_image_frame = -1  # 用於同步感測器
		self.image_queue = queue.Queue()  # 新增 queue

	# Renamed cleanup to _destroy_actors and improved its logic
	def _destroy_actors(self):
		for actor in self.actor_list:
			if actor is not None and actor.is_alive:
				if hasattr(actor, 'is_listening') and actor.is_listening:
					actor.stop()
				actor.destroy()
		self.actor_list = []
	
	def maintain_speed(self,s):
			''' 
			this is a very simple function to maintan desired speed
			s arg is actual current speed
			'''
			if s >= self.PREFERRED_SPEED:
				return 0
			elif s < self.PREFERRED_SPEED - self.SPEED_THRESHOLD:
				return 0.7 # think of it as % of "full gas"
			else:
				return 0.3 # tweak this if the car is way over or under preferred speed 
			
	def apply_cnn(self,im):
		img = np.float32(im)
		img = img /255
		img = np.expand_dims(img, axis=0)
		cnn_applied = self.cnn_model([img,0],training=False)
		cnn_applied = np.squeeze(cnn_applied)
		return  cnn_applied ##[0][0]
	def step(self, action):
		self.step_counter += 1
		steer = action[0]
		# --- 離散steer mapping ---
		if steer == 0:
			steer = -0.9
		elif steer == 1:
			steer = -0.25
		elif steer == 2:
			steer = -0.1
		elif steer == 3:
			steer = -0.05
		elif steer == 4:
			steer = 0.0
		elif steer == 5:
			steer = 0.05
		elif steer == 6:
			steer = 0.1
		elif steer == 7:
			steer = 0.25
		elif steer == 8:
			steer = 0.9
		# --- end mapping ---
		v = self.vehicle.get_velocity()
		kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
		estimated_throttle = self.maintain_speed(kmh)
		self.vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer, brake=0.0))
		self.world.tick()
		# 用 queue 取影像
		try:
			cam = self.image_queue.get(timeout=2.0)  # 最多等2秒
		except queue.Empty:
			print('[step] Warning: sensor image not updated after 2s')
			cam = self.front_camera  # fallback
		distance_travelled = self.initial_location.distance(self.vehicle.get_location())
		# storing camera to return at the end in case the clean-up function destroys it
		cam = self.front_camera
		# showing image
		if self.SHOW_CAM:
			cv2.imshow('Sem Camera', cam)
			cv2.waitKey(1)

		# track steering lock duration to prevent "chasing its tail"
		lock_duration = 0
		if self.steering_lock == False:
			if steer<-0.6 or steer>0.6:
				self.steering_lock = True
				self.steering_lock_start = time.time()
		else:
			if steer<-0.6 or steer>0.6:
				lock_duration = time.time() - self.steering_lock_start
		
		# start defining reward from each step
		reward = 0
		done = False
		#punish for collision
		if len(self.collision_hist) != 0:
			done = True
			reward = reward - 300
			# self.cleanup() # Call to cleanup removed
		if len(self.lane_invade_hist) != 0:
			done = True
			reward = reward - 250
			# self.cleanup() # Call to cleanup removed
		# punish for steer lock up
		if lock_duration>3:
			done = True
			reward = reward - 150
			# self.cleanup() # Call to cleanup removed
		elif lock_duration > 1:
			reward = reward - 20
		#reward for acceleration
		#if kmh < 10:
		#	reward = reward - 3
		#elif kmh <15:
		#	reward = reward -1
		#elif kmh>40:
		#	reward = reward - 10 #punish for going to fast
		#else:
		#	reward = reward + 1
		# reward for making distance
		if distance_travelled<30:
			reward = reward - 1
		elif distance_travelled<50:
			reward =  reward + 1
		else:
			reward = reward + 2
		# check for episode duration
		if self.episode_start + SECONDS_PER_EPISODE < time.time():
			done = True
			# self.cleanup() # Call to cleanup removed
		self.image_for_CNN = self.apply_cnn(self.front_camera[self.height_from:,self.width_from:self.width_to])
		return self.image_for_CNN, reward, done, {}	#curly brackets - empty dictionary required by SB3 format

	def reset(self):
		self._destroy_actors() # Call _destroy_actors at the beginning of reset

		self.collision_hist = []
		self.lane_invade_hist = []
		# self.actor_list = [] # Already handled by _destroy_actors
		self.transform = random.choice(self.world.get_map().get_spawn_points())
		
		self.vehicle = None
		while self.vehicle is None:
			try:
        # connect
				self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
			except:
				pass
		self.actor_list.append(self.vehicle)
		self.initial_location = self.vehicle.get_location()
		self.sem_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
		self.sem_cam.set_attribute("image_size_x", f"{self.im_width}")
		self.sem_cam.set_attribute("image_size_y", f"{self.im_height}")
		self.sem_cam.set_attribute("fov", f"90")
		
		camera_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z,x=self.CAMERA_POS_X))
		self.sensor = self.world.spawn_actor(self.sem_cam, camera_init_trans, attach_to=self.vehicle)
		self.actor_list.append(self.sensor)
		self._last_image_frame = -1
		def _process_img(image):
			image.convert(carla.ColorConverter.CityScapesPalette)
			i = np.array(image.raw_data)
			i = i.reshape((self.im_height, self.im_width, 4))[:, :, :3]
			self.front_camera = i
			self._last_image_frame = image.frame
			try:
				self.image_queue.put_nowait(i)
			except queue.Full:
				pass  # 若 queue 滿則略過
		self.sensor.listen(_process_img)
		self.image_queue.queue.clear()  # reset 時清空 queue

		self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
		self.world.tick()  # 確保所有 actor/sensor 都同步
		time.sleep(0.1)
        # now apply random yaw so the RL does not guess to go straight

		angle_adj = random.randrange(-SPIN, SPIN, 1)
		trans = self.vehicle.get_transform()
		trans.rotation.yaw = trans.rotation.yaw + angle_adj
		self.vehicle.set_transform(trans)

		# showing camera at the spawn point
		if self.SHOW_CAM:
			cv2.namedWindow('Sem Camera',cv2.WINDOW_AUTOSIZE)
			cv2.imshow('Sem Camera', self.front_camera)
			cv2.waitKey(1)
		colsensor = self.blueprint_library.find("sensor.other.collision")
		self.colsensor = self.world.spawn_actor(colsensor, camera_init_trans, attach_to=self.vehicle)
		self.actor_list.append(self.colsensor)
		self.colsensor.listen(lambda event: self.collision_data(event))

		lanesensor = self.blueprint_library.find("sensor.other.lane_invasion")
		self.lanesensor = self.world.spawn_actor(lanesensor, camera_init_trans, attach_to=self.vehicle)
		self.actor_list.append(self.lanesensor)
		self.lanesensor.listen(lambda event: self.lane_data(event))

		# 等待第一張影像
		wait_count = 0
		while self.front_camera is None or self._last_image_frame == -1:
			self.world.tick()
			time.sleep(0.01)
			wait_count += 1
			if wait_count > 100:
				print('[reset] Warning: sensor image not received after 100 ticks')
				break
		self.episode_start = time.time()
		self.steering_lock = False
		self.steering_lock_start = None # this is to count time in steering lock and start penalising for long time in steering lock
		self.step_counter = 0
		self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
		self.image_for_CNN = self.apply_cnn(self.front_camera[self.height_from:,self.width_from:self.width_to])
		return self.image_for_CNN

	def process_img(self, image):
		image.convert(carla.ColorConverter.CityScapesPalette)
		i = np.array(image.raw_data)
		i = i.reshape((self.im_height, self.im_width, 4))[:, :, :3] # this is to ignore the 4th Alpha channel - up to 3
		self.front_camera = i

	def collision_data(self, event):
		self.collision_hist.append(event)
	def lane_data(self, event):
		self.lane_invade_hist.append(event)

	def close(self):
		self._destroy_actors()
		if self.SHOW_CAM:
			cv2.destroyAllWindows()