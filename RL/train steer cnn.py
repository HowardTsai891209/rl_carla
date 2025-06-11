'''
This is using cnn pre-trained layer to pre-process the image
'''


from stable_baselines3 import PPO #PPO
from typing import Callable
import os
from carenv_steer_only_cnn import CarEnv
import time
from stable_baselines3.common.callbacks import BaseCallback # Added import

TIMESTEPS = 20000 # how long is each training iteration - individual steps
iters = 0
iters_limit = 120 # how many training iterations you want

# Added PrintTimestepsCallback class
class PrintTimestepsCallback(BaseCallback):
    def __init__(self, print_freq=1000, total_timesteps_per_iteration=(TIMESTEPS * iters_limit), verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.total_timesteps_per_iteration = total_timesteps_per_iteration
        self.iteration_timesteps = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.iteration_timesteps = 0 # Reset for each model.learn() call if reset_num_timesteps=True
                                     # If reset_num_timesteps=False, this callback might need adjustment
                                     # or be re-instantiated per iteration for clear per-iteration progress.
                                     # For this case, we assume it's instantiated per iteration.

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0: # self.num_timesteps is the total steps for this learn() call
            percent = 100 * self.num_timesteps / self.total_timesteps_per_iteration
            print(f"[Train Steer CNN] Iteration Timesteps: {self.num_timesteps}/{self.total_timesteps_per_iteration} ({percent:.1f}%)")
        return True

print('This is the start of training script')

print('setting folders for logs and models')
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

print('connecting to env..')

env = CarEnv()

env.reset()
print('Env has been reset as part of launch')
model = PPO('MlpPolicy', env, verbose=1,learning_rate=0.001, tensorboard_log=logdir)

# Instantiate callback for total progress
progress_callback = PrintTimestepsCallback(print_freq=1000, total_timesteps_per_iteration=TIMESTEPS*iters_limit)
while iters<iters_limit:  # how many training iterations you want
	iters += 1
	print('Iteration ', iters,' is to commence...')
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO_Iter{iters}", callback=progress_callback )
	print('Iteration ', iters,' has been trained')
	model.save(f"{models_dir}/{TIMESTEPS*iters}")