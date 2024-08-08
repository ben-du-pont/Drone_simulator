from pathlib import Path
from UWB_sim_RL_environment import UWBAnchorEnv
from stable_baselines3 import PPO

# Prepare directories for saving models and logs
package_path = Path(__file__).parent.resolve()
models_dir = package_path / 'models'
logdir = package_path / 'logs'

models_dir.mkdir(exist_ok=True)
logdir.mkdir(exist_ok=True)

model_path = models_dir / 'ppo_uwb_anchor_100000'

env_id = 'UWBAnchor-v0'
env_kwargs = {'render_mode': 'human'}
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