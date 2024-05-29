import gymnasium as gym
from stable_baselines3 import PPO
from UWB_sim_RL_environment import UWBAnchorEnv
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
import os
import time
from pathlib import Path
import torch

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

# Prepare directories for saving models and logs
package_path = Path(__file__).parent.resolve()
models_dir = package_path / 'models'
logdir = package_path / 'logs'
models_dir.mkdir(exist_ok=True)
logdir.mkdir(exist_ok=True)


# Environment setup
env_id = 'UWBAnchor-v0'
env = make_vec_env(env_id, n_envs=4)  # Using vectorized environments to speed up training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PPO('MlpPolicy', env, verbose=1, device=device, tensorboard_log=logdir, n_steps=4096, batch_size=256, n_epochs=20)

# Training parameters
TIMESTEPS = 1e6
total_iterations = 10  # Set a total number of iterations for training
iters = 0

# Training loop
while iters < total_iterations:
    iters += 1  # Increment the iteration counter
    print(f"Starting iteration {iters}...")
    
    # Train the model for a specified number of timesteps
    model.learn(total_timesteps=TIMESTEPS, tb_log_name="PPO")
    
    # Save the model after each iteration
    model.save(models_dir / f"ppo_uwb_anchor_{TIMESTEPS * iters}")

    print(f"Model saved after iteration {iters}")

print("Training complete.")

# Load and evaluate the model
print("Loading the model...")
model_path = models_dir / f"ppo_uwb_anchor_{TIMESTEPS * total_iterations}"
model = PPO.load(model_path)
print("Model loaded.")

env = UWBAnchorEnv()
print("Evaluating the model...")
obs, info = env.reset()
for i in range(10):
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    print(f"Step {i}: Action: {action}, Reward: {rewards}, Done: {dones}, Truncated: {truncated}, Info: {info}")
    env.render()

print("Evaluation complete.")
