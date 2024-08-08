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
env_kwargs = {'render_mode': 'human'}
env = make_vec_env(env_id, n_envs=1, env_kwargs=env_kwargs)  # Using vectorized environments to speed up training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = models_dir / 'ppo_uwb_anchor_latest'
model_path = models_dir / 'ppo_uwb_anchor_new.zip'

ppo_params = {
    'n_steps': 4096,
    'batch_size': 128,
    'n_epochs': 10,
    'learning_rate': 1e-5,
    'ent_coef': 0.01,
    'clip_range': 0.2,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5
}

if model_path.exists():
    print("Loading the existing model...")
    old_model = PPO.load(model_path, env=env, device=device)
    print("Model loaded.")
    
    # Extract the policy weights
    policy_state_dict = old_model.policy.state_dict()
    
    # Create a new model with the updated parameters
    model = PPO('MlpPolicy', env, verbose=1, device=device, tensorboard_log=logdir, **ppo_params)
    
    # Load the policy weights into the new model
    model.policy.load_state_dict(policy_state_dict)

else:
    print("Creating a new model...")

    model = PPO('MlpPolicy', env, verbose=1, device=device, tensorboard_log=logdir, **ppo_params)

# Training parameters
TIMESTEPS = 1e5
total_iterations = 100  # Set a total number of iterations for training
iters = 0


# Training loop
while iters < total_iterations:
    iters += 1  # Increment the iteration counter
    print(f"Starting iteration {iters}...")
    
    # Train the model for a specified number of timesteps
    model.learn(total_timesteps=TIMESTEPS, tb_log_name="PPO")
    
    # Save the model after each iteration
    model.save(models_dir / "ppo_uwb_anchor_new")

    print(f"Model saved after iteration {iters}")

    # obs = env.reset()
    # action, _states = model.predict(obs)
    # obs, rewards, dones, truncated = env.step(action)
    # print(f"Action: {action}, Reward: {rewards}, Done: {dones}, Truncated: {truncated}")
    # env.render()

print("Training complete.")




# Load and evaluate the model
print("Loading the model...")
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
