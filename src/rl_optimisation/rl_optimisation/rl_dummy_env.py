import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import VecNormalize

# Custom environment for initial training
class MimicObservationEnv(gym.Env):
    def __init__(self):
        super(MimicObservationEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(18,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(15,), dtype=np.float32)

    def reset(self, seed=None):
        self.state = 2*(np.random.rand(18) - np.tile([0.5], 18))  # Initialize state with random values
        return self.state, {}

    def step(self, action):
        target = self.state[-15:]  # Last 15 elements of the observation
        reward = -np.mean((action - target) ** 2)  # Negative MSE as reward
        done = False
        self.state = 2*(np.random.rand(18) - np.tile([0.5], 18))  # Update state
        return self.state, reward, done, done, {}




env = MimicObservationEnv()
print("Evaluating the model...")

model = PPO.load("ppo_mimic_observation")

obs, info = env.reset()

for i in range(10):
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    print(f"Step {i}: Observation: {obs} Action: {action}, Reward: {rewards}, Done: {dones}, Truncated: {truncated}, Info: {info}")



# Create and wrap the environment
env = MimicObservationEnv()

# Initialize the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=1000000)  # Adjust the timesteps as needed

# Save the model
model.save("ppo_mimic_observation")


